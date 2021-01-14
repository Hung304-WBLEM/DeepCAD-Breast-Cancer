import json

def read(json_path):
	with open(json_path) as f:
		data = json.load(f)
	return data
