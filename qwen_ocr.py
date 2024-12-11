from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

from PIL import Image
import requests

image = Image.open("image.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "Explain me about the image in details."
            }
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text = [text_prompt],
    images = [image],
    padding = True,
    return_tensors = "pt"
)

inputs = inputs.to("cuda")

output_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

print(output_text)

image = Image.open("Simple-Invoice-Template.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "Please make a table of the information available in the image with minute details. DONOT ADD ANY FURTHER DETAILS AND DONOT REPAEAT."
            }
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text = [text_prompt],
    images = [image],
    padding = True,
    return_tensors = "pt"
)

inputs = inputs.to("cuda")

output_ids = model.generate(**inputs, max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

print(output_text)

output_text

