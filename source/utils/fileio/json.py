import json

def read(json_path):
	with open(json_path) as f:
		data = json.load(f)
	return data


def write(data, json_path):
        with open(json_path, 'w') as f:
                json.dump(data, f)
