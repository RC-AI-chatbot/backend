import typesense
import json
import os
import requests
import psycopg2
from fastapi import APIRouter, Body
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/typesense", tags=["typesense"])

TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "")
TYPESENSE_PORT = os.getenv("TYPESENSE_PORT", "")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY","") 
FEED_URL = os.getenv("FEED_URL", "")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "csv")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

client = typesense.Client({
    "nodes": [{
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": "https"
    }],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 1000
})

def fetch_parts_catalog(cursor):
    cursor.execute("""
        SELECT 
            id, item_group_id, sku, part_category, part_description, 
            part_parentsku_compatibility, part_product_group_code, part_type
        FROM parts_catalog
    """)
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    parts = []
    for row in rows:
        part = dict(zip(cols, row))
        for k, v in part.items():
            part[k] = str(v) if v is not None else ''
        part["compat_ids"] = [id_.strip() for id_ in part["part_parentsku_compatibility"].split(",") if id_.strip()]
        parts.append(part)
    return parts

@router.post("/sync_feeddata")
async def sync_data_from_bigcommerce() -> str:
    """
        Sync data from remote feed into Typesense
    """
    
    feed_response = requests.get(FEED_URL)
    feed_response.raise_for_status()
    feed = feed_response.json()

    products = feed.get("products", [])

    # 2. Fetch parts catalog from PostgreSQL
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )

    cursor = conn.cursor()
    parts_catalog = fetch_parts_catalog(cursor)

    schema_fields = [
        "id", "title", "description", "brand", "sku", "parent_sku", "mpn", "gtin",
        "item_group_id", "availability", "product_category_deprecated", "model_type",
        "model_skill_level", "model_speed", "color", "price", "sale_price", "price_range",
        "shipping_weight", "image_link", "link", "date_created", "view_count", "total_sold"
    ]

    int_fields = ["view_count", "item_group_id"]

    enriched_products = []
    for product in products:
        item_group_id = str(product.get("item_group_id", ""))
        compatible_parts = [
            {k: v for k, v in part.items() if k != "compat_ids"}
            for part in parts_catalog if item_group_id in part["compat_ids"]
        ]
        product['compatible_parts'] = compatible_parts
        # Ensure all required fields are present and types are correct
        product['id'] = str(product.get('id', ''))
        product['mpn'] = str(product.get('mpn', ''))
        product['model_skill_level'] = str(product.get('model_skill_level', ''))
        product['gtin'] = str(product.get('gtin', ''))
        product['total_sold'] = str(product.get('total_sold', ''))

        for field in schema_fields:
            value = product.get(field)
            if value is None or value == "":
                product[field] = ""
            else:
                product[field] = str(value)

        # Handle int fields
        for field in int_fields:
            value = product.get(field)
            try:
                product[field] = int(value)
            except (TypeError, ValueError):
                product[field] = 0

        enriched_products.append(product)

    print(products[1])

    # 4. Write enriched products to JSONL
    temp_jsonl_path = "temp_products.jsonl"
    with open(temp_jsonl_path, "w", encoding="utf-8") as f:
        for product in enriched_products:
            f.write(json.dumps(product, ensure_ascii=False) + "\n")

    schema = {
        "name": "slot_cars",
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "title", "type": "string"},
            {"name": "description", "type": "string"},
            {"name": "brand", "type": "string"},
            {"name": "sku", "type": "string"},
            {"name": "parent_sku", "type": "string"},
            {"name": "mpn", "type": "string"},
            {"name": "gtin", "type": "string"},
            {"name": "item_group_id", "type": "int32"},
            {"name": "availability", "type": "string"},
            {"name": "product_category_deprecated", "type": "string"},
            {"name": "model_type", "type": "string"},
            {"name": "model_skill_level", "type": "string"},
            {"name": "model_speed", "type": "string"},
            {"name": "color", "type": "string"},
            {"name": "price", "type": "string"},
            {"name": "sale_price", "type": "string"},
            {"name": "price_range", "type": "string"},
            {"name": "shipping_weight", "type": "string"},
            {"name": "image_link", "type": "string"},
            {"name": "link", "type": "string"},
            {"name": "date_created", "type": "string"},
            {"name": "view_count", "type": "int32"},
            {"name": "total_sold", "type": "string"},
            {"name": "compatible_parts", "type": "string[]"}
        ],
        "default_sorting_field": "view_count"
    }

    # Delete old collection if it exists
    try:
        client.collections['slot_cars'].delete()
    except Exception:
        pass

    client.collections.create(schema)

    # 5. Read the JSONL file and import to Typesense
    with open(temp_jsonl_path, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()]

    results = client.collections['slot_cars'].documents.import_(
        docs,
        {'action': 'upsert'}
    )

    # Optionally, remove the temp file
    os.remove(temp_jsonl_path)

    print(results)
    return "Sync completed"