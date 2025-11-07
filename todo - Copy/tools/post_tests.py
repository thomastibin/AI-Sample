import time
import httpx


def post(msg):
    url = 'http://localhost:8001/chat'
    for i in range(10):
        try:
            r = httpx.post(url, json={'message': msg}, timeout=30.0)
            print('STATUS:', r.status_code)
            print('BODY:', r.text)
            return r
        except Exception as e:
            print('wait, retrying...', i, str(e))
            time.sleep(0.5)
    print('failed to reach server')


print('--- message 1: create meeting ---')
post('create a meeting with thomastibin@gmail.com on next week tuesday 7pm')
print('\n--- message 2: find meeting ---')
post('find the meeting with thomastibin@gmail.com')
