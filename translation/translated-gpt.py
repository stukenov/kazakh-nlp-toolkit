from transformers import pipeline
import threading
import ipywidgets as widgets

# Initialize the pipeline for text generation with the local model
# Set the device to 0 to run on GPU with limited memory
generator = pipeline('text-generation', model='EleutherAI/gpt-j-6b', device=0, framework='pt', use_fast=False)

# Create a text input widget
text_input = widgets.Textarea(
    description='Input Text:',
    style={'description_width': 'initial'}
)

# Create a button to send request to LLM
send_request_button = widgets.Button(
    description='Send Request',
    button_style='success',
    tooltip='Click to send request to LLM'
)

# Create an output widget to display the response from LLM
output = widgets.Output()

def generate_response(input_text):
    # Generate text using the model
    # Set no_repeat_ngram_size to 2 and use do_sample to reduce memory usage
    # Adjust max_length to fit the memory constraints
    generated = generator(input_text, max_length=30, pad_token_id=50256, no_repeat_ngram_size=2, do_sample=True, temperature=0.7)
    return generated[0]['generated_text']

def on_send_request_button_clicked(b):
    # When the button is clicked, generate the response and display it
    with output:
        output.clear_output()
        try:
            response_text = generate_response(text_input.value.strip())
            print(response_text)
        except Exception as e:
            print(f"An error occurred: {e}")

# Link the button click event to the function
send_request_button.on_click(on_send_request_button_clicked)

# Display the widgets
display(text_input, send_request_button, output)
