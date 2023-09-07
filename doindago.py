from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('himateitsniceday.html')

@app.route('/your_server_endpoint', methods=['POST'])
def process_text(): ## 여기서 text가 감지된 텍스트니까 이 텍스트가지고 모델에 넣어보면 될듯. 가져올때마다 모델에 넣으면 너무 부하가 많이 걸릴것같으니까 주기로 가져오거나 그래야될듯?
    data = request.get_json()
    text = data['text']
    print('감지된 텍스트:', text)
    return '텍스트가 성공적으로 전송되었습니다.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
