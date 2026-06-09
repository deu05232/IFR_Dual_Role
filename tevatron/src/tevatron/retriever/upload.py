from huggingface_hub import upload_folder, create_repo



repo_id="repo_id"

create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
    private=False,
)

upload_folder(
    repo_id=repo_id,
    folder_path="folder_dir",
    path_in_repo=".",
)



from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Slack API Token (OAuth Token을 입력)
SLACK_TOKEN = "slack_token_here"

# Slack 클라이언트 생성
client = WebClient(token=SLACK_TOKEN)

def send_slack_message(channel, text):
    """Slack 특정 채널에 메시지를 전송하는 함수"""
    try:
        response = client.chat_postMessage(channel=channel, text=text)
        print("Slack 메시지 전송 성공!")
    except SlackApiError as e:
        print(f"Slack 메시지 전송 실패: {e.response['error']}")
        

send_slack_message("channel_id_here", "upload 완료!")