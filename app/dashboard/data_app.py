import os

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

DEFAULT_DATABASE_URL = "mysql+aiomysql://uritomo_user:uritomo_pass@localhost:3306/uritomo"


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("mysql+aiomysql://"):
        return raw_url.replace("mysql+aiomysql://", "mysql+pymysql://", 1)
    if raw_url.startswith("mysql+asyncmy://"):
        return raw_url.replace("mysql+asyncmy://", "mysql+pymysql://", 1)
    if raw_url.startswith("mysql://"):
        return raw_url.replace("mysql://", "mysql+pymysql://", 1)
    return raw_url


def get_database_url() -> str:
    load_dotenv()
    raw_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("database_url")
        or os.getenv("SQLALCHEMY_DATABASE_URI")
        or DEFAULT_DATABASE_URL
    )
    return _normalize_database_url(raw_url)


def load_tables(engine) -> list[str]:
    inspector = inspect(engine)
    return sorted(inspector.get_table_names())


def fetch_all_rows(engine, table_name: str) -> list[dict]:
    query = text(f"SELECT * FROM `{table_name}`")
    with engine.connect() as connection:
        result = connection.execute(query)
        return list(result.mappings().all())


def fetch_users(engine) -> list[dict]:
    query = text("SELECT id, display_name, email FROM `users` ORDER BY display_name, id")
    with engine.connect() as connection:
        result = connection.execute(query)
        return list(result.mappings().all())


def load_user_foreign_keys(engine) -> dict[str, list[dict]]:
    inspector = inspect(engine)
    table_fks: dict[str, list[dict]] = {}
    for table_name in inspector.get_table_names():
        fks = inspector.get_foreign_keys(table_name)
        user_fks = []
        for fk in fks:
            if fk.get("referred_table") != "users":
                continue
            columns = fk.get("constrained_columns") or []
            if len(columns) != 1:
                continue
            user_fks.append(
                {
                    "column": columns[0],
                    "referred_columns": fk.get("referred_columns") or ["id"],
                }
            )
        if user_fks:
            table_fks[table_name] = user_fks
    return table_fks


def fetch_fk_counts(engine, table_name: str, column_name: str) -> dict[str, int]:
    query = text(
        f"SELECT `{column_name}` AS user_id, COUNT(*) AS cnt "
        f"FROM `{table_name}` "
        f"WHERE `{column_name}` IS NOT NULL "
        f"GROUP BY `{column_name}`"
    )
    with engine.connect() as connection:
        result = connection.execute(query)
        return {row["user_id"]: row["cnt"] for row in result.mappings().all()}


def build_user_relationship_rows(
    users: list[dict], table_fks: dict[str, list[dict]], fk_counts
) -> list[dict]:
    rows: list[dict] = []
    for user in users:
        user_id = user["id"]
        display_name = user.get("display_name") or "Unknown"
        email = user.get("email") or ""

        for table_name, fks in table_fks.items():
            for fk in fks:
                column = fk["column"]
                cnt = fk_counts.get((table_name, column), {}).get(user_id, 0)
                if cnt <= 0:
                    continue
                rows.append(
                    {
                        "user_id": user_id,
                        "display_name": display_name,
                        "email": email,
                        "table": table_name,
                        "column": column,
                        "count": cnt,
                    }
                )
    return rows


def main() -> None:
    st.set_page_config(page_title="Data", layout="wide")
    st.title("MySQL Data Viewer")
    st.caption("Displays every table and all rows in the configured database.")

    db_url = get_database_url()

    try:
        engine = create_engine(db_url, pool_pre_ping=True)
    except Exception as exc:  # pragma: no cover - runtime guard
        st.error("Failed to create database engine.")
        st.code(str(exc))
        st.stop()

    tables = load_tables(engine)
    if not tables:
        st.info("No tables found in the database.")
        return

    st.header("User Relationships (Table View)")
    if "users" not in tables:
        st.info("No users table found in the database.")
    else:
        try:
            users = fetch_users(engine)
        except Exception as exc:  # pragma: no cover - runtime guard
            st.error("Failed to load users.")
            st.code(str(exc))
            users = []

        if not users:
            st.info("No users found in the database.")
        else:
            table_fks = load_user_foreign_keys(engine)
            if not table_fks:
                st.info("No tables referencing users were found.")
            else:
                fk_counts = {}
                for table_name, fks in table_fks.items():
                    for fk in fks:
                        fk_counts[(table_name, fk["column"])] = fetch_fk_counts(
                            engine, table_name, fk["column"]
                        )

                rows = build_user_relationship_rows(users, table_fks, fk_counts)
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No user-related records found.")

    st.header("Tables")
    for table_name in tables:
        st.subheader(f"Table: {table_name}")
        try:
            rows = fetch_all_rows(engine, table_name)
        except Exception as exc:  # pragma: no cover - runtime guard
            st.error(f"Failed to load table: {table_name}")
            st.code(str(exc))
            continue

        st.caption(f"{len(rows)} rows")
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.write("No rows in this table.")


if __name__ == "__main__":
    main()
