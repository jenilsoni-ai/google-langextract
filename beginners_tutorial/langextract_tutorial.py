import langextract as lx
import textwrap

api_key="your_api_key"

# Define the extraction task in a clear, detailed prompt.
prompt = textwrap.dedent("""
    Extract key entities and their relationships from financial news.
    The goal is to map a knowledge graph of companies, people, and events.
    Entities to extract: Company, Person, Event.
    Relationships to extract: CEO_of, Acquired, Launched.
    Use exact text for extraction. Do not paraphrase.
    For relationships, include 'source_entity' and 'target_entity' in the attributes.
""")

# Create a few-shot example to guide the model on the desired output format.
# This example is crucial for getting accurate and consistently structured data.
example_text = (
    "In a major move, Tesla acquired a small AI startup named DeepDrive. "
    "Elon Musk, the CEO of Tesla, celebrated the deal in a tweet."
)
examples = [
    lx.data.ExampleData(
        text=example_text,
        extractions=[
            lx.data.Extraction(
                extraction_class="Company",
                extraction_text="Tesla"
            ),
            lx.data.Extraction(
                extraction_class="Company",
                extraction_text="DeepDrive"
            ),
            lx.data.Extraction(
                extraction_class="Person",
                extraction_text="Elon Musk"
            ),
            lx.data.Extraction(
                extraction_class="Event",
                extraction_text="Tesla acquired a small AI startup named DeepDrive",
                attributes={"event_type": "Acquisition"}
            ),
            lx.data.Extraction(
                extraction_class="Relationship",
                extraction_text="acquired",
                attributes={
                    "type": "Acquired",
                    "source_entity": "Tesla",
                    "target_entity": "DeepDrive"
                }
            ),
            lx.data.Extraction(
                extraction_class="Relationship",
                extraction_text="CEO",
                attributes={
                    "type": "CEO_of",
                    "source_entity": "Elon Musk",
                    "target_entity": "Tesla"
                }
            ),
        ],
    )
]

# The text you want to analyze.
input_text = (
    "Alphabet's subsidiary, Google, is preparing to launch a new "
    "AI-powered search tool. Sundar Pichai, the CEO of Google, "
    "announced the new product at a press event."
)

# Run the extraction using a Gemini model.
try:
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id="gemini-2.5-flash",
        api_key=api_key 
    )

    # Print the raw extracted data. This data is what you would use to build the graph.
    print(f"Extracted data for: '{input_text}'\n")
    print(result)
    
    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

    # Generate the visualization from the file
    html_content = lx.visualize("extraction_results.jsonl")
    with open("visualization.html", "w", encoding="utf-8") as f:
        f.write(html_content)

except Exception as e:
    print(f"An error occurred during extraction: {e}")