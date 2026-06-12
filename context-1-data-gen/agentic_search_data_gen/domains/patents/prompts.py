# extract from non final rejection
NON_FINAL_REJECTION_EXTRACTION_SYSTEM_PROMPT = "You are a helpful assistant that extracts components from a non final rejection to a patent application."

NON_FINAL_REJECTION_EXTRACTION_PROMPT = """You are a patent analysis assistant. Your task is to parse a USPTO non-final office action rejection and extract structured data about each claim rejection.

## Input
You will be given:
1. The claims of the rejected patent application
2. The non-final rejection office action text

## Task
For each claim rejected under 35 U.S.C. §102 or §103, extract and structure the rejection details.

### Key Definitions
- **Rejection Type**: Either "102" (anticipation - single reference) or "103" (obviousness - one or more references combined)
- **Claim Element**: A specific limitation or feature from the claim text
- **Prior Art Element**: The corresponding feature in the prior art (only applicable to the text citations) that the examiner maps to the claim element
- **Citation**: The specific location in the prior art (abstract, paragraph numbers, etc.)
- **Reasoning**: The examiner's rationale for why prior art teaches the claim element

### Extraction Rules
1. Create ONE entry per unique combination of: claim number + rejection type + prior art set
2. If the same claim is rejected under both §102 and §103 (using different art), create separate entries for each
3. **ONLY extract claim elements that have text citations** (e.g., "Abstract", "para. 0066", "para. 0071-0077"). Skip any elements where the examiner only cites figures or other non-text references without accompanying text citations. If a claim element has both text citations and figure citations, only extract the parts corresponding to the text citations. Abstracts count as text citations.
4. Capture which elements come from which reference
5. Include the examiner's combination rationale/motivation in the reasoning field
6. If examiner uses words like "implied," "known," "understood," or "obvious," preserve this language in prior_art_element to flag weak mappings
7. For dependent claims, include the parent claim as well in the claim_text, so the rejection is self-contained
8. If no §102 or §103 rejections exist in the office action, output only: None
9. If a rejection has no claim elements with text citations, exclude the rejection entry
10. If the claim rejection includes a failed teaching, such as "Regarding claim 2, Niimi teaches the limitations of claim 1 above but fails t o teach a thickness o f
the coating layer is 5 nm to 50 nm, and optionally, the thickness of the coating layer is 10 nm t o 20 nm. Instead Niimi teaches coating a carbonaceous substance onto a surface of the lithium silicate particles in a thickness o f from 100 nm t o 1,000 nm is possible (Niimi [0063]).", exclude this from the claim element mappings. We should only extract the successful teachings.

## Output Format

If no §102 or §103 rejections are found, output exactly:
None

Otherwise, return your response as valid XML following the structure below. Do not include any text outside the XML tags.

### Format Template
<rejections>
  <rejection>
    <type>102 or 103</type>
    <claim_number>claim number</claim_number>
    <claim_text>full text of the claim, including parent claim if applicable</claim_text>
    <claim_element_mappings>
      <mapping>
        <claim_element>element of claim being rejected</claim_element>
        <prior_art_element>what examiner says teaches this</prior_art_element>
        <citations>
          <citation>
            <name>reference short name</name>
            <locations>abstract, para. numbers, etc. - comma separated</locations>
          </citation>
        </citations>
        <mapping_strength>explicit, implied, or argued</mapping_strength>
      </mapping>
      <!-- additional mappings for each claim element -->
    </claim_element_mappings>
    <reasoning>examiner's rationale</reasoning>
  </rejection>
  <rejection>
    <!-- next rejection -->
  </rejection>
  <!-- additional rejections as needed -->
</rejections>

### Field Definitions

| Field | Description |
|-------|-------------|
| type | "102" or "103" |
| claim_number | The claim number as a string |
| claim_text | Complete claim text from the application |
| name | Short name examiner uses for the reference/citation |
| patent_number | Full patent/publication number with country code |
| claim_element | The specific limitation text from the claim |
| prior_art_element | What the examiner asserts teaches this element (preserve examiner's language) |
| locations | Paragraph citations only (e.g., "para. 0066", "¶0071-0077", "[0042]") |
| mapping_strength | "explicit" (directly stated), "implied" (examiner infers), "argued" (examiner argues equivalence) |
| reasoning | Examiner's reasoning for why prior art teaches the claim element |

## Example

### Input Claims (excerpt):
CLAIMS
What is claimed is:
A wireless smartphone device comprising:
a housing portion; the housing portion comprising:
	one or more cameras;
		one or more microphones configured to receive and transmit an audio signal;
		one or more biometric sensors;
		one or more environmental sensors;
		a holographic display system; and
	a display; and
an earbud portion disposed on and extending from a surface of the housing portion, the earbud portion configured for insertion into the ear canal of a user.

The wireless smartphone device of claim 1, wherein the one or more biometric sensors comprises at least one of a pulse oximeter, heart rate monitor, thermometer, and motion sensor.

The wireless smartphone device of claim 2, wherein the wireless smartphone device is configured to analyze data collected from the one or more biometric sensors and determine a user's emotional state.

...

### Input Rejection (excerpt):
...
Claims 1-2, and 5 are rejected under 35 U.S.C. 103 as being unpatentable
and obvious over Candy et al (US 20190327355, A1) in view o f Osterhout et al.
(US 20150309316, A1).
Regarding claim 1, Candy teaches a wireless smartphone device (at least the
Abstract, Figs. 1-5, and para. 0071-0077 teaches FONCHP devices 1 namely a
smart earphone device including an earpiece and earphones incorporating all the
features of a smartphone and/or mobile electronic tablet, said device as described
further in para. 0071 comprises said wireless smartphone device), comprising:
Application/Control Number: 19/312,195
Art Unit: 2682
one or more cameras (Fig. 1, camera 4);
Page 7
one or more microphones configured to receive and transmit an audio signal (Fig.
1, microphone 6);
one or more biometric sensors (the at least microscopic sensors of para. 0066,
furthermore, the camera 4 o f at least fig. 1 and/or the microphone 6 both further
include a t least one or more known biometric sensors);
one or more environmental sensors (the at least weather received signals o f further
para. 0073 further comprise signals received by at least one o r more implied
environmental sensors, secondly, the camera 4 of at least fig. 1 and/or the
microphone 6 both further include a t least one or more known environmental
sensors);
a holographic display system; and
a display (display 44 of a t least Fig. 5); and
an earbud portion disposed on and extending from a surface of the housing portion,
the earbud portion configured for insertion into the ear canal o f a user (earbud 2 of
at least fig. 1 comprising said portion disposed on and extending from a surface of
the housing portion, the earbud portion configured for insertion into the ear canal
o f a user).
Candy is silent regarding the above lined-out items such as a holographic
display system.
Application/Control Number: 19/312,195
Art Unit: 2682
Page 8
Osterhout teaches a smart glasses device 100 of at least Fig. 1 and 7 comprises a
wireless smartphone capability and an earbud portion 120 disposed on and
extending from a surface o f the housing portion, the earbud portion configured for
insertion into the ear canal o f a user, smart wireless device 100 further comprises a
display screen o f Fig. 7, and further in at least para. 0222 and 0467 a holographic
display system. It would have been obvious to one of ordinary skill in the art
before the effective filing date o f the claimed invention to combine the teachings of
Candy i n view o f Osterhout to include wherein said wireless smartphone device
comprising a holographic display system, as discussed above, as Candy in view o f
Osterhout are in the same field o f endeavor o f providing a hands-free smart device,
mobile phone, wearable further configured with known means to perform multiple
functions in addition to being attached to an earbud portion disposed on and
extending from a surface o f the housing portion, the earbud portion configured for
insertion into the ear canal of a user, Osterhout's smart glasses incorporating a
smartphone accompanied by the holographic display system complements the
smart earphone o f Candy with said corresponding holographic display system i n
the sense that the architecture o f Osterhout when combined with the architecture o f
the smart earphone of Candy provides a smart earphone device with the same
capability of a smartphone but with a design and utility providing 3D display
imaging capability to the user further increasing visual focus according to further
Application/Control Number: 19/312,195
Art Unit: 2682
Page 9
known methods to yield predictable results since known work in one field o f
endeavor may prompt variations of it for use in either the same field or a different
one based on design incentives or other market forces i f the variations are
predictable to one of ordinary skill in the art as said combination is thus the
adaptation o f an old idea or invention using newer technology that is either
commonly available and understood in the art thereby a variation on already
known art (See MPEP 2143, KSR Exemplary Rationale F).
Regarding claim 2 (according to claim 1), Candy further teaches wherein the one
o r more biometric sensors comprises at least one o f a pulse oximeter, heart rate
monitor, thermometer, and motion sensor (at least para. 0013, 0073, and 0077
further teaches device 1/11 further provided with meteorological features to inform
and make the User aware o f surrounding environment weather conditions detected
by at least a thermometer or the like).

...

### Output:
<rejections>
  <rejection>
    <type>103</type>
    <claim_number>1</claim_number>
    <claim_text>A wireless smartphone device comprising: a housing portion; the housing portion comprising: one or more cameras; one or more microphones configured to receive and transmit an audio signal; one or more biometric sensors; one or more environmental sensors; a holographic display system; and a display; and an earbud portion disposed on and extending from a surface of the housing portion, the earbud portion configured for insertion into the ear canal of a user.</claim_text>
    <claim_element_mappings>
      <mapping>
        <claim_element>a wireless smartphone device</claim_element>
        <prior_art_element>FONCHP devices 1 is a smart earphone device including an earpiece and earphones incorporating all the features of a smartphone and/or mobile electronic tablet</prior_art_element>
        <citations>
          <citation>
            <name>Candy</name>
            <locations>Abstract, para. 0071-0077</locations>
          </citation>
        </citations>
        <mapping_strength>explicit</mapping_strength>
      </mapping>
      <mapping>
        <claim_element>one or more biometric sensors</claim_element>
        <prior_art_element>microscopic sensors</prior_art_element>
        <citations>
          <citation>
            <name>Candy</name>
            <locations>para. 0066</locations>
          </citation>
        </citations>
        <mapping_strength>argued</mapping_strength>
      </mapping>
      <mapping>
        <claim_element>one or more environmental sensors</claim_element>
        <prior_art_element>weather received signals received by "implied environmental sensors"</prior_art_element>
        <citations>
          <citation>
            <name>Candy</name>
            <locations>para. 0073</locations>
          </citation>
        </citations>
        <mapping_strength>implied</mapping_strength>
      </mapping>
      <mapping>
        <claim_element>a holographic display system</claim_element>
        <prior_art_element>holographic display system in smart glasses</prior_art_element>
        <citations>
          <citation>
            <name>Osterhout</name>
            <locations>para. 0222, 0467</locations>
          </citation>
        </citations>
        <mapping_strength>explicit</mapping_strength>
      </mapping>
    </claim_element_mappings>
    <reasoning>Candy's smart earphone has everything except holographic display. Osterhout's smart glasses have holographic display. Combine them because they're both wearables with smartphone features, and you get the claimed invention.</reasoning>
  </rejection>
  <rejection>
    <type>103</type>
    <claim_number>2</claim_number>
    <claim_text>The wireless smartphone device of claim 1 (A wireless smartphone device comprising:
a housing portion; the housing portion comprising:
	one or more cameras;
		one or more microphones configured to receive and transmit an audio signal;
		one or more biometric sensors;
		one or more environmental sensors;
		a holographic display system; and
	a display; and
an earbud portion disposed on and extending from a surface of the housing portion, the earbud portion configured for insertion into the ear canal of a user.), wherein the one or more biometric sensors comprises at least one of a pulse oximeter, heart rate monitor, thermometer, and motion sensor.</claim_text>
    ...
  </rejection>
  <!-- additional rejections for other claims following same structure -->
</rejections>

## Your Task

### Patent Claims:
{rejected_patent_claims}

### Non-Final Rejection:
{non_final_rejection}

## Instructions
1. Read through the entire rejection carefully
2. Identify each distinct rejection (unique claim + rejection type + prior art combination)
3. For each rejection, extract all required fields following the format template
4. Pay special attention to examiner language indicating weak mappings ("implied," "known," "understood," "obvious to include")
5. Distinguish between primary and secondary references in §103 rejections
6. If no §102 or §103 rejections exist, output only: None
7. Otherwise, output valid XML only — no additional commentary or explanation before or after

Output your response following the format specified above.
"""

# create task from non final rejection
EVAL_GEN_SYSTEM_PROMPT = "You are a helpful assistant that rephrases rejections from patent examiners to create a search task for prior art."

EVAL_GEN_PROMPT = """
Your task is to analyze a patent claim rejection and rephrase the rejection into a search task for prior art retrieval. The goal is to create a challenging retrieval task that tests a system's ability to find relevant prior art.

## Rules
**MUST obfuscate (replace with synonyms/paraphrases):**
- Any terms from the prior art abstracts that do NOT appear in the rejected claim
- Names, patent numbers, acronyms, or other identifiers
- Technical jargon unique to the prior art description

**Obfuscation techniques to use:**
- Replace specific terms with broader category descriptions (e.g., "earpiece" → "auricular apparatus")
- Use functional descriptions instead of component names (e.g., "microscopic sensors" → "sensing elements")
- Substitute domain-specific jargon with plain language equivalents

## Output Format

Refer to each cited prior art as "Prior Art A", "Prior Art B", etc. (in order of appearance).

Output your response in the following format, no other text or formatting:
<task>
{{task}}
</task>

## Style Rules

Write the task as a straightforward description of what each prior art document contains.

**Do NOT use patent examination language such as:**
- "teaches", "discloses", "is relied upon", "limitation", "one skilled in the art", "would have been obvious", "in view of", "the combination of these references"

**Instead, use plain descriptive language such as:**
- "describes", "contains", "includes", "covers", "addresses", "is about", "features"

The output should read like a description of documents to find, not like a patent office action.

## Example

### Input
Rejected claim:
A wireless smartphone device comprising: a housing portion; the housing portion comprising: one or more cameras; one or more microphones configured to receive and transmit an audio signal; one or more biometric sensors; one or more environmental sensors; a holographic display system; and a display; and an earbud portion disposed on and extending from a surface of the housing portion, the earbud portion configured for insertion into the ear canal of a user.

Rejection details:
Citations: Candy
Claim element: 'a wireless smartphone device'
Prior art element: 'FONCHP devices 1 is a smart earphone device including an earpiece and earphones incorporating all the features of a smartphone and/or mobile electronic tablet'
Mapping strength: 'explicit'

---
Citations: Candy
Claim element: 'one or more biometric sensors'
Prior art element: 'microscopic sensors; furthermore, the camera and/or microphone both further include at least one or more known biometric sensors'
Mapping strength: 'argued'

---
Citations: Candy
Claim element: 'one or more environmental sensors'
Prior art element: 'weather received signals further comprise signals received by at least one or more implied environmental sensors'
Mapping strength: 'implied'

---
Citations: Osterhout
Claim element: 'a holographic display system'
Prior art element: 'holographic display system'
Mapping strength: 'explicit'

---
Rejection reasoning:
Candy teaches a smart earphone device with smartphone features but is silent on holographic display system. Osterhout teaches smart glasses with wireless smartphone capability and holographic display system. It would have been obvious to combine because both are in the same field of endeavor of providing hands-free smart devices with known means to perform multiple functions. Osterhout's holographic display system complements Candy's smart earphone providing 3D display imaging capability according to known methods to yield predictable results (MPEP 2143, KSR Exemplary Rationale F).

Prior Art Abstracts:
Candy: Abstract The Free On Call Phone (FONCHP) devices are mobile style telecommunications mechanisms which allows a user to make and receive telephone calls in a hands-free manner. The FONCHP devices comprise an earpiece and incorporates all the features of a smartphone and/or mobile electronic tablet but without the need for using one's hands. The FONCHP devices are voice sound activated and controlled by voice prompts, commands given by the user to which the FONCHP devices are programmed to respond.

Osterhout: Abstract This disclosure concerns an interactive head-mounted eyepiece with an integrated processor for handling content for display and an integrated image source for introducing the content to an optical assembly through which the user views a surrounding environment and the displayed content, wherein the eyepiece includes predictive control of external device based on an event input.

### Output
<task>
This claim is rejected based on Prior Art A and Prior Art B. Prior Art A describes a telecommunications device that attaches to the auricular apparatus, which addresses the claimed wireless communication device through its integrated mobile capabilities, and satisfies the biometric detection requirement through sensing elements. Environmental monitoring is implicitly taught through the device's ability to process atmospheric condition data. Prior Art B teaches an optical head-worn computing system with AR capabilities. Prior Art B is relied upon to address the holographic display system, which Prior Art A does not disclose.
</task>

## Input
Rejected claim:
{rejected_patent_claim}

Rejection details:
{rejection_details}

Rejection reasoning:
{rejection_reasoning}

Prior Art Abstracts:
{prior_art_abstracts}

Output the search task in the format specified above, no other text or formatting.
"""
