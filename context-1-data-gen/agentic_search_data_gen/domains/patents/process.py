import os
import json
from dotenv import load_dotenv
from docx import Document
import pandas as pd
from io import BytesIO
from pathlib import Path
from datalab_sdk import AsyncDatalabClient, ConvertOptions
import asyncio
import argparse
import httpx
import tempfile

load_dotenv()

TARGET_CODE = 'CTNF' # non final rejection
CODES_TO_EXTRACT = ['CTNF', 'CLM', 'SPEC', 'ABST', '892']

class PatentDataProcessor:
    def __init__(self):
        self.uspto_api_key = os.getenv("USPTO_API_KEY")
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.datalab_api_key = os.getenv("DATALAB_API_KEY")
        self.processed_application_numbers = set()
        self.errors = []  # Track errors with reasons

        # Connection pools (initialized lazily)
        self._uspto_client = None  # httpx.AsyncClient for USPTO API
        self._search_client = None  # httpx.AsyncClient for search API
        self._datalab_client = None  # AsyncDatalabClient for PDF processing

    async def get_uspto_client(self):
        """Get or create the shared USPTO async client."""
        if self._uspto_client is None:
            self._uspto_client = httpx.AsyncClient(
                timeout=30.0,
                headers={'X-API-KEY': self.uspto_api_key},
                http2=True,
                follow_redirects=True
            )
        return self._uspto_client

    async def get_search_client(self):
        """Get or create the shared search API async client."""
        if self._search_client is None:
            self._search_client = httpx.AsyncClient(
                timeout=30.0,
                http2=True,
            )
        return self._search_client

    async def get_datalab_client(self):
        """Get or create the shared datalab client."""
        if self._datalab_client is None:
            self._datalab_client = await AsyncDatalabClient(api_key=self.datalab_api_key).__aenter__()
        return self._datalab_client

    async def close(self):
        """Close all connection pools."""
        if self._uspto_client is not None:
            await self._uspto_client.aclose()
            self._uspto_client = None
        if self._search_client is not None:
            await self._search_client.aclose()
            self._search_client = None
        if self._datalab_client is not None:
            await self._datalab_client.__aexit__(None, None, None)
            self._datalab_client = None

    def get_latest_docs_before_target(self, df, target_code, codes_to_extract):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True)

        target_rows = df[df['document_code'] == target_code].sort_values('date')

        if target_rows.empty:
            raise ValueError(f"No rows found with document_code '{target_code}'")

        first_occurrence_date = target_rows['date'].iloc[0]

        filtered_df = df[df['date'] <= first_occurrence_date]

        result_df = (
            filtered_df[filtered_df['document_code'].isin(codes_to_extract)]
            .sort_values('date', ascending=False)
            .drop_duplicates(subset='document_code', keep='first')
        )

        return result_df

    async def extract_patent_data(self, application_number):
        try:
            url = f'https://api.uspto.gov/api/v1/patent/applications/{application_number}/documents'
            client = await self.get_uspto_client()
            response = (await client.get(url)).json()

            docs = response['documentBag']

            def get_download_url(download_options, doc_code):
                if doc_code == '892':
                    for item in download_options:
                        if item['mimeTypeIdentifier'] == 'PDF':
                            return item['downloadUrl']
                else:
                    for item in download_options:
                        if item['mimeTypeIdentifier'] == 'MS_WORD':
                            return item['downloadUrl']

                    for item in download_options:
                        if item['mimeTypeIdentifier'] == 'PDF':
                            return item['downloadUrl']

                return None

            times = [item['officialDate'] for item in docs]
            document_codes = [item['documentCode'] for item in docs]
            document_ids = [item['documentIdentifier'] for item in docs]
            download_options = [item['downloadOptionBag'] for item in docs]

            full_df = pd.DataFrame({
                'document_code': document_codes,
                'document_id': document_ids,
                'date': times,
                'download_options': download_options
            })

            extracted_df = self.get_latest_docs_before_target(full_df, TARGET_CODE, CODES_TO_EXTRACT)

            all_docs = {}

            for i, row in extracted_df.iterrows():
                download_url = get_download_url(row['download_options'], row['document_code'])

                if download_url is None:
                    continue

                all_docs[row['document_code']] = {
                    'document_id': row['document_id'],
                    'download_url': download_url
                }

            if len(all_docs) < len(CODES_TO_EXTRACT):
                return None, "Failed to extract all required documents (CTNF, CLM, SPEC, ABST, 892)"

            return all_docs, None

        except ValueError:
            return None, "No CTNF (non-final rejection) document found before target date"
        except Exception:
            return None, "Failed to retrieve or parse patent data from USPTO API"

    async def get_response(self, url):
        client = await self.get_uspto_client()
        return await client.get(url)

    async def docx_url_to_text(self, url):
        response = await self.get_response(url)

        doc = Document(BytesIO(response.content))

        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(full_text)

    async def pdf_url_to_text(self, url):
        response = await self.get_response(url)

        # Use unique temp file to avoid race conditions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(response.content)
            pdf_path = Path(f.name)

        try:
            client = await self.get_datalab_client()
            result = await client.convert(pdf_path)
            return result.markdown
        finally:
            pdf_path.unlink(missing_ok=True)

    async def extract_references(self, url):
        response = await self.get_response(url)

        # Use unique temp file to avoid race conditions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(response.content)
            pdf_path = Path(f.name)

        schema = {
            "type": "object",
            "properties": {
                "referenced_patents": {
                    "type": "array",
                    "description": "A list of patent document numbers listed. Include both US and foreign patents.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_number": {
                                "type": "string",
                                "description": "The document number of the patent. (i.e. US-20170114017-A1)"
                            },
                            "author_name": {
                                "type": "string",
                                "description": "The name of the author of the patent as it appears in the cell. (i.e. Vanlerber; Guy)"
                            }
                        },
                        "required": [
                            "document_number",
                            "author_name"
                        ]
                    }
                }
            },
            "required": ["referenced_patents"]
        }

        options = ConvertOptions(
            page_schema=json.dumps(schema),
            mode="balanced"
        )

        try:
            client = await self.get_datalab_client()
            result = await client.convert(pdf_path, options=options)
            extracted_result = json.loads(result.extraction_schema_json)
            return [{'document_number': item['document_number'], 'author_name': item['author_name']} for item in extracted_result['referenced_patents']]
        finally:
            pdf_path.unlink(missing_ok=True)

    async def get_reference_info(self, pub_number):
        try:
            url = "https://www.searchapi.io/api/v1/search"
            params = {
                "engine": "google_patents_details",
                "patent_id": f"patent/{pub_number}/en",
                "api_key": self.search_api_key
            }

            client = await self.get_search_client()
            response = await client.get(url, params=params)
            response_dict = response.json()

            title = response_dict['title']
            inventors = [item['name'] for item in response_dict['inventors']]
            application_number = ''.join(c for c in response_dict['application_number'] if c.isdigit())
            abstract = response_dict['abstract']
            spec = response_dict['description']
            claims = response_dict['claims']
            similar_patents = [item['publication_number'] for item in response_dict['similar_documents'] if 'publication_number' in item]

            return {
                'application_number': application_number,
                'title': title,
                'inventors': inventors,
                'abstract': abstract,
                'spec': spec,
                'claims': claims,
                'similar_patents': similar_patents
            }
        except:
            return None

    async def process_patent(self, application_number):
        if application_number in self.processed_application_numbers:
            self.errors.append({
                'application_number': application_number,
                'reason': "Application number already exists in dataset"
            })
            return None

        self.processed_application_numbers.add(application_number)

        patent_mapping, error = await self.extract_patent_data(application_number)

        if patent_mapping is None:
            self.errors.append({
                'application_number': application_number,
                'reason': error
            })
            return None

        patent_mapping['application_number'] = application_number

        try:
            # Process all documents in parallel
            async def process_document(doc_code, doc_info):
                download_url = doc_info['download_url']

                if doc_code == '892':
                    references = await self.extract_references(download_url)
                    return doc_code, 'references', references
                else:
                    if download_url.endswith('.docx'):
                        text = await self.docx_url_to_text(download_url)
                    elif download_url.endswith('.pdf'):
                        text = await self.pdf_url_to_text(download_url)
                    else:
                        text = None
                    return doc_code, 'text', text

            doc_tasks = [
                process_document(doc_code, doc_info)
                for doc_code, doc_info in patent_mapping.items()
                if doc_code != 'application_number'
            ]
            doc_results = await asyncio.gather(*doc_tasks)

            # Apply results and process references
            references_to_process = []
            for doc_code, result_type, result in doc_results:
                if result_type == 'text':
                    patent_mapping[doc_code]['text'] = result
                elif result_type == 'references':
                    patent_mapping[doc_code]['references'] = {}
                    references_to_process = result

            # Process all references in parallel
            if references_to_process:
                async def process_reference(reference):
                    ref_doc_number = reference['document_number'].replace('-', '')
                    ref_author_name = reference['author_name']
                    reference_info = await self.get_reference_info(ref_doc_number)
                    return ref_doc_number, ref_author_name, reference_info

                reference_tasks = [process_reference(ref) for ref in references_to_process]
                reference_results = await asyncio.gather(*reference_tasks)

                for ref_doc_number, ref_author_name, reference_info in reference_results:
                    if reference_info is not None:
                        ref_application_number = reference_info['application_number']
                        if ref_application_number in self.processed_application_numbers:
                            self.errors.append({
                                'application_number': ref_application_number,
                                'reason': "Application number already exists in dataset (found as reference)"
                            })
                            continue

                        self.processed_application_numbers.add(ref_application_number)

                        patent_mapping['892']['references'][ref_doc_number] = reference_info
                        patent_mapping['892']['references'][ref_doc_number]['extracted_author_name'] = ref_author_name
                        patent_mapping['892']['references'][ref_doc_number]['application_number'] = ref_application_number

            patent_mapping['similar_patents'] = {}

            # Depth 1: Get similar patents from references
            if '892' in patent_mapping and len(patent_mapping['892']['references']) > 0:
                # Collect all unique similar patents
                similar_patents_to_fetch = []
                for k, v in patent_mapping['892']['references'].items():
                    for similar_patent in v['similar_patents']:
                        if similar_patent not in patent_mapping['similar_patents']:
                            similar_patents_to_fetch.append(similar_patent)

                # Remove duplicates
                similar_patents_to_fetch = list(set(similar_patents_to_fetch))

                # Fetch all similar patents in parallel
                async def fetch_similar_patent(similar_patent):
                    similar_patent_info = await self.get_reference_info(similar_patent.replace('-', ''))
                    return similar_patent, similar_patent_info

                if similar_patents_to_fetch:
                    similar_tasks = [fetch_similar_patent(sp) for sp in similar_patents_to_fetch]
                    similar_results = await asyncio.gather(*similar_tasks)

                    for similar_patent, similar_patent_info in similar_results:
                        if similar_patent_info is not None:
                            similar_app_num = similar_patent_info['application_number']
                            if similar_app_num not in self.processed_application_numbers:
                                self.processed_application_numbers.add(similar_app_num)
                                patent_mapping['similar_patents'][similar_patent] = similar_patent_info
                            else:
                                self.errors.append({
                                    'application_number': similar_app_num,
                                    'reason': "Application number already exists in dataset (found as similar patent)"
                                })

            return patent_mapping

        except Exception:
            self.errors.append({
                'application_number': application_number,
                'reason': "Failed to process and extract document contents (text/PDF/references)"
            })
            return None

    def load_existing_patents(self, output_dir):
        """Scan output directory and populate processed_application_numbers from existing files."""
        output_path = Path(output_dir)
        if not output_path.exists():
            return

        json_files = list(output_path.glob('*.json'))
        # Exclude errors.json
        json_files = [f for f in json_files if f.name != 'errors.json']

        print(f"Found {len(json_files)} existing patent files in {output_dir}")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    patent_data = json.load(f)

                # Add main patent's application number
                if 'application_number' in patent_data:
                    self.processed_application_numbers.add(patent_data['application_number'])

                # Add application numbers from references
                if '892' in patent_data and 'references' in patent_data['892']:
                    for ref_key, ref_data in patent_data['892']['references'].items():
                        if 'application_number' in ref_data:
                            self.processed_application_numbers.add(ref_data['application_number'])

                # Add application numbers from similar patents
                if 'similar_patents' in patent_data:
                    for sim_key, sim_data in patent_data['similar_patents'].items():
                        if 'application_number' in sim_data:
                            self.processed_application_numbers.add(sim_data['application_number'])

            except Exception as e:
                print(f"Warning: Could not load {json_file.name}: {e}")

        print(f"Loaded {len(self.processed_application_numbers)} existing application numbers")

    async def process_multiple_patents(self, application_numbers, output_dir='sample_subset', max_concurrent=10):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load existing patents to avoid reprocessing
        self.load_existing_patents(output_dir)

        # Semaphore to limit concurrent requests (helps with rate limiting)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(app_num):
            """Process a single patent with semaphore control, returning app_num with result"""
            async with semaphore:
                result = await self.process_patent(app_num)
                return app_num, result

        # Create tasks for all patents with concurrency control
        tasks = [process_with_semaphore(app_num) for app_num in application_numbers]

        # Process patents and save results as they complete
        completed = 0
        total = len(application_numbers)
        successful = 0

        for coro in asyncio.as_completed(tasks):
            try:
                app_num, result = await coro

                if result is None:
                    print(f"[{completed + 1}/{total}] Skipped {app_num}: See errors.json for details")
                else:
                    output_path = Path(output_dir) / f"{app_num}.json"
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=4)
                    print(f"[{completed + 1}/{total}] Saved: {output_path}")
                    successful += 1

            except Exception as e:
                self.errors.append({
                    'application_number': 'unknown',
                    'reason': "Unexpected error during async processing"
                })
                print(f"[{completed + 1}/{total}] Error processing patent: {e}")

            completed += 1

        # Save errors to file
        errors_path = Path(output_dir) / "errors.json"
        with open(errors_path, 'w') as f:
            json.dump(self.errors, f, indent=4)

        failed = total - successful
        print(f"\nProcessing Summary:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        if self.errors:
            print(f"  Errors saved to: {errors_path}")


def read_application_numbers(file_path):
    """Read application numbers from a text file (one per line)."""
    with open(file_path, 'r') as f:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]


async def main():
    """CLI entry point for processing patents from command line."""
    parser = argparse.ArgumentParser(
        description='Process multiple patent applications concurrently',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_data.py --input applications.txt
  python process_data.py --input applications.txt --output my_output --concurrent 5
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to text file containing application numbers (one per line)'
    )
    parser.add_argument(
        '--output', '-o',
        default='sample_subset',
        help='Output directory for JSON files (default: sample_subset)'
    )
    parser.add_argument(
        '--concurrent', '-c',
        type=int,
        default=10,
        help='Maximum concurrent requests (default: 10)'
    )

    args = parser.parse_args()

    # Read application numbers
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return

    application_numbers = read_application_numbers(input_file)
    print(f"Found {len(application_numbers)} application numbers to process")
    print(f"Output directory: {args.output}")
    print(f"Max concurrent requests: {args.concurrent}")
    print("-" * 60)

    # Process patents
    processor = PatentDataProcessor()
    try:
        await processor.process_multiple_patents(
            application_numbers,
            output_dir=args.output,
            max_concurrent=args.concurrent
        )
    finally:
        await processor.close()

    print("-" * 60)
    print("Processing complete!")


if __name__ == '__main__':
    asyncio.run(main())
