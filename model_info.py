import argparse
from pathlib import Path

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, OpenAI


def main():
    parser = argparse.ArgumentParser(description="OpenAI 모델 목록과 호출 가능 여부 확인")
    parser.add_argument("--text", metavar="MODEL_ID", help="Responses API로 지정한 모델 한 번 호출")
    parser.add_argument("--check-embeddings", action="store_true", help="목록의 임베딩 후보를 한 번씩 호출")
    args = parser.parse_args()

    # 터미널 위치와 관계없이 이 파일 옆의 .env를 읽습니다.
    env_file = Path(__file__).with_name(".env")
    api_key = (dotenv_values(env_file).get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print(f"OPENAI_API_KEY를 확인하세요. 설정 파일 {env_file}")
        return 1

    # 오류 발생 시 SDK의 자동 재시도를 끕니다.
    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                    max_retries=0, timeout=30.0)
    try:
        model_ids = sorted(model.id for model in client.models.list())
    except APIStatusError as error:
        print(f"모델 목록 조회 실패 (HTTP {error.status_code}). 키와 모델 목록 조회 권한을 확인하세요.")
        return 1
    except APIConnectionError:
        print("모델 목록 조회 실패. 네트워크 연결을 확인하세요.")
        return 1

    # 목록 API에는 모델 종류 필드가 없어서 공개 임베딩 모델의 이름을 기준으로 찾습니다.
    embedding_ids = [name for name in model_ids if name.startswith("text-embedding-")]
    print(f"모델 목록 ({len(model_ids)}개)")
    for model_id in model_ids:
        print(f"  {model_id}")

    print("\n임베딩 후보 (호출 전)")
    for model_id in embedding_ids:
        print(f"  {model_id}")
    if not embedding_ids:
        print("  목록에 text-embedding-으로 시작하는 모델이 없습니다.")

    if args.text:
        try:
            response = client.responses.create(
                model=args.text, input="Reply with OK.",
                max_output_tokens=64, store=False,
            )
            if response.output_text.strip():
                print(f"\n텍스트 생성 확인  {args.text}")
            else:
                print(f"\n요청은 수락됐지만 텍스트가 비어 있음  {args.text} (상태 {response.status})")
        except APIStatusError as error:
            print(f"\n텍스트 호출 미확인  {args.text} (HTTP {error.status_code})")
        except APIConnectionError:
            print(f"\n텍스트 호출 미확인  {args.text} (네트워크 오류)")

    if args.check_embeddings:
        print("\n임베딩 실제 호출 결과")
        for model_id in embedding_ids:
            try:
                result = client.embeddings.create(model=model_id, input="test")
                dimensions = len(result.data[0].embedding)
                print(f"  생성 확인  {model_id} ({dimensions}차원)")
            except APIStatusError as error:
                print(f"  호출 미확인  {model_id} (HTTP {error.status_code})")
            except APIConnectionError:
                print(f"  호출 미확인  {model_id} (네트워크 오류)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
