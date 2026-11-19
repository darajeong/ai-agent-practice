import os
import sys
import psycopg
from psycopg import sql
from dotenv import load_dotenv

load_dotenv()
ROWS = [
    ("notice-01", "과제 제출", "이번 주 과제 제출 기한은 금요일 오후 6시입니다.", [1,0,0]),
    ("notice-02", "실습 준비", "API Key는 로컬 .env 파일에만 저장합니다.", [0,1,0]),
    ("notice-03", "질문 응대", "과제 관련 문의는 게시판에 남겨 주세요.", [0,0,1]),
]
QUESTIONS = {
    "마감일": [1,0,0],
    "API Key 관리": [0,1,0],
    "문의 게시판": [0,0,1],
}
MODEL = "text-embedding-3-small"
def connect():
    for key in ("DB_NAME", "DB_USER", "DB_PASSWORD"):
        if not os.getenv(key):
            raise ValueError(f".env 항목 누락: {key}")
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=5,
    )
def vec(values):
    return "[" + ",".join(str(float(x)) for x in values) + "]"
def embed(texts):
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("여기에"):
        raise ValueError("Live 모드에는 실제 OPENAI_API_KEY가 필요합니다.")
    if os.getenv("OPENAI_EMBEDDING_MODEL", MODEL) != MODEL:
        raise ValueError("이 예제는 기본 1536차원 모델만 지원합니다.")
    from openai import OpenAI
    response = OpenAI(api_key=key).embeddings.create(model=MODEL, input=texts)
    result = [x.embedding for x in sorted(response.data, key=lambda x: x.index)]
    if len(result) != len(texts) or any(len(v) != 1536 for v in result):
        raise ValueError("임베딩 개수/차원이 일치하지 않습니다.")
    return result
def setup(cur):
    for table, dims in (("lesson06_fixture_docs", 3), ("lesson06_live_docs", 1536)):
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
              id TEXT PRIMARY KEY, content TEXT NOT NULL,
              section TEXT NOT NULL, source TEXT NOT NULL,
              embedding vector({}) NOT NULL
            )""").format(sql.Identifier(table), sql.SQL(str(dims))))
def save(cur, table, records):
    assert table in ("lesson06_fixture_docs", "lesson06_live_docs")
    statement = sql.SQL("""
        INSERT INTO {} (id, section, content, source, embedding)
        VALUES (%s,%s,%s,%s,%s::vector)
        ON CONFLICT (id) DO UPDATE SET
          section=EXCLUDED.section, content=EXCLUDED.content,
          source=EXCLUDED.source, embedding=EXCLUDED.embedding
    """).format(sql.Identifier(table))
    for id_, section, content, vector in records:
        cur.execute(statement, (id_, section, content, "lesson04_notice.md", vec(vector)))
def search(cur, table, vector):
    assert table in ("lesson06_fixture_docs", "lesson06_live_docs")
    statement = sql.SQL("""
        SELECT id, section, content, source, embedding <=> %s::vector AS distance
        FROM {} ORDER BY embedding <=> %s::vector, id LIMIT 3
    """).format(sql.Identifier(table))
    cur.execute(statement, (vec(vector), vec(vector)))
    for id_, section, content, source, distance in cur.fetchall():
        print(f"[RESULT] id={id_}, section={section}, distance={distance:.4f}")
        print(f"         source={source}, content={content}")
def run():
    if len(sys.argv) < 2:
        raise ValueError("명령: check|index|count|query|live-index|live-query")
    cmd = sys.argv[1]
    if cmd not in ("check", "index", "count", "query", "live-index", "live-query"):
        raise ValueError(f"지원하지 않는 명령: {cmd}")
    if cmd in ("query", "live-query") and len(sys.argv) != 3:
        raise ValueError("질문을 따옴표로 감싸서 입력하세요.")
    with connect() as conn:
        with conn.cursor() as cur:
            if cmd == "check":
                cur.execute("SELECT current_database(), current_user")
                database, user = cur.fetchone()
                cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
                version = cur.fetchone()
                print(f"[DB] connected: {database}, user={user}")
                print(f"[PGVECTOR] {version[0] if version else 'NOT ENABLED'}")
                if not version:
                    raise RuntimeError("STEP 02: CREATE EXTENSION vector가 필요합니다.")
                return
            setup(cur)
            if cmd == "index":
                save(cur, "lesson06_fixture_docs", ROWS)
                print("[INDEX] fixture 3건; API 호출 0회")
            elif cmd == "count":
                for table in ("lesson06_fixture_docs", "lesson06_live_docs"):
                    cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
                    print(f"[COUNT] {table}: {cur.fetchone()[0]}")
            elif cmd == "query":
                question = sys.argv[2]
                if question not in QUESTIONS:
                    raise ValueError("Fixture 질문은 마감일, API Key 관리, 문의 게시판 중 선택")
                search(cur, "lesson06_fixture_docs", QUESTIONS[question])
            elif cmd == "live-index":
                vectors = embed([row[2] for row in ROWS])
                save(cur, "lesson06_live_docs", [
                    (row[0], row[1], row[2], v) for row, v in zip(ROWS, vectors)
                ])
                print("[INDEX] live 3건; API 호출 발생")
            else:
                search(cur, "lesson06_live_docs", embed([sys.argv[2]])[0])
if __name__ == "__main__":
    try:
        run()
    except (ValueError, RuntimeError, psycopg.Error) as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)