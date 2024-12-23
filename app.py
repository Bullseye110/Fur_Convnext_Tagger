import gradio as gr
import huggingface_hub
from PIL import Image
from pathlib import Path
import onnxruntime as rt
import numpy as np
import csv
import spaces
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

# Download and setup model
e621_model_path = Path(huggingface_hub.snapshot_download('toynya/Z3D-E621-Convnext'))
# Only use CPU provider since CUDA might not be available
e621_model_session = rt.InferenceSession(e621_model_path / 'model.onnx', providers=["CPUExecutionProvider"])
with open(e621_model_path / 'tags-selected.csv', mode='r', encoding='utf-8') as file:
	csv_reader = csv.DictReader(file)
	e621_model_tags = [row['name'].strip() for row in csv_reader]

# Create a thread-local storage for ONNX sessions
thread_local = threading.local()

def get_session():
	if not hasattr(thread_local, "session"):
		thread_local.session = rt.InferenceSession(
			e621_model_path / 'model.onnx',
			providers=["CPUExecutionProvider"]
		)
	return thread_local.session

def prepare_image_e621(image: Image.Image, target_size: int):
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to numpy array
	image_array = np.asarray(padded_image, dtype=np.float32)
	# Convert PIL-native RGB to BGR
	image_array = image_array[:, :, ::-1]

	return np.expand_dims(image_array, axis=0)

def process_single_image(img_file):
	try:
		img = Image.open(img_file.name).convert('RGB')
		image_array = prepare_image_e621(img, 448)
		
		session = get_session()
		input_name = 'input_1:0'
		output_name = 'predictions_sigmoid'
		
		result = session.run([output_name], {input_name: image_array})
		result = result[0][0]
		
		scores = {e621_model_tags[i]: float(result[i]) for i in range(len(result))}
		predicted_tags = [tag for tag, score in scores.items() if score > 0.3]
		tag_string = ', '.join(predicted_tags).replace("_", " ")
		
		return tag_string
	except Exception as e:
		return f"Error processing {Path(img_file.name).name}: {str(e)}"

def predict_e621(image: Image.Image):
	THRESHOLD = 0.3
	image_array = prepare_image_e621(image, 448)

	input_name = 'input_1:0'
	output_name = 'predictions_sigmoid'

	result = e621_model_session.run([output_name], {input_name: image_array})
	result = result[0][0]

	scores = {e621_model_tags[i]: float(result[i]) for i in range(len(result))}
	predicted_tags = [tag for tag, score in scores.items() if score > THRESHOLD]
	tag_string = ', '.join(predicted_tags).replace("_", " ")

	return tag_string, scores

def predict_batch_e621(files, progress=gr.Progress(track_tqdm=True)):
	if not files:
		return "No images provided"
	
	total_files = len(files)
	results = [""] * total_files
	result_queue = Queue()
	
	def process_and_queue(idx, file):
		result = process_single_image(file)
		result_queue.put((idx, result))
	
	# Start processing files in parallel
	with ThreadPoolExecutor(max_workers=4) as executor:
		# Submit all tasks
		futures = [
			executor.submit(process_and_queue, idx, file)
			for idx, file in enumerate(files)
		]
		
		# Update results as they complete
		completed = 0
		while completed < total_files:
			idx, result = result_queue.get()
			results[idx] = result
			completed += 1
			
			# Update progress
			progress(completed / total_files, f"Processing image {completed}/{total_files}")
			
			# Yield current results
			yield "\n\n".join(r for r in results if r)
	
	# Return final results
	return "\n\n".join(r for r in results if r)

DESCRIPTION = """
E621 Tagger (Z3D-E621-Convnext) 
- Image => E621 Pony Prompt
- Mod of [fancyfeast's demo](https://huggingface.co/spaces/fancyfeast/Z3D-E621-Convnext-space) for toynya's [Z3D-E621-Convnext](https://huggingface.co/toynya/Z3D-E621-Convnext)
"""

# Single image interface
single_image_interface = gr.Interface(
	predict_e621,
	inputs=gr.Image(label="Source", type='pil'),
	outputs=[
		gr.Textbox(label="Tag String", show_copy_button=True),
		gr.Label(label="Tag Predictions", num_top_classes=100),
	],
	description=DESCRIPTION,
	allow_flagging="never",
)

# Batch processing interface
batch_interface = gr.Interface(
	predict_batch_e621,
	inputs=gr.File(label="Upload Images", file_count="multiple", file_types=["image"]),
	outputs=gr.Textbox(label="Batch Results", show_copy_button=True, lines=10),
	description="Batch Processing - Drop multiple images here",
	allow_flagging="never",
)

# Create tabbed interface
demo = gr.TabbedInterface(
	[single_image_interface, batch_interface],
	["Single Image", "Batch Processing"]
)

if __name__ == '__main__':
	# Launch with queue enabled for batch processing
	demo.queue(concurrency_count=1, max_size=20).launch(share=True)
