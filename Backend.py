
from flask import Flask, request, jsonify,send_file
import os

app = Flask(__name__)
@app.route('/tts_start', methods=['GET'])
def process_data():
	print("oss")
	return '成功'

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True,port=10001)