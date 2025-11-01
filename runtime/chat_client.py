import requests, sys, json

def main():
    if len(sys.argv) < 3 or sys.argv[1] != '--prompt':
        print('Usage: python runtime/chat_client.py --prompt "Your message"')
        sys.exit(1)

    prompt = sys.argv[2]
    url = 'http://localhost:5000'
    headers = {'Content-Type': 'application/json'}
    payload = {'prompt': prompt}

    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        r.raise_for_status()
        print('Server response:', r.json().get('response'))
    except Exception as e:
        print('Error contacting server:', e)

if __name__ == '__main__':
    main()