import boto3
import json
from boto3.dynamodb.conditions import Key
import os
import datetime
import re
import smtplib
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from decimal import Decimal

# Initialize DynamoDB with better error handling
try:
    dynamodb = boto3.resource('dynamodb', region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    table = dynamodb.Table('test_genie_sg')  # Updated table name
    
    # Test the connection by checking if table exists
    table.load()
    print(f"Successfully connected to DynamoDB table: {table.table_name}")
except Exception as e:
    print(f"Error connecting to DynamoDB: {str(e)}")
    # Removed st.error to hide red error box


def get_chat_histories():
    """Get all chat histories sorted by last_updated"""
    try:
        response = table.scan()
        histories = response.get('Items', [])
        sorted_histories = sorted(histories, key=lambda x: x.get('last_updated', 0), reverse=True)
        print(f"Retrieved {len(sorted_histories)} chat histories")
        return sorted_histories
    except Exception as e:
        print(f"Error retrieving chat histories: {str(e)}")
        # Removed st.error to hide red error box
        return []


def get_chat_history(session_id):
    """Get chat history for a specific session"""
    try:
        print(f"Retrieving chat history for session: {session_id}")
        response = table.query(KeyConditionExpression=Key('session_id').eq(session_id))
        items = response.get('Items', [])
        
        if items and 'chat_history' in items[0] and items[0]['chat_history']:
            chat_data = items[0]['chat_history']
            
            # Handle both string and list formats
            if isinstance(chat_data, str):
                try:
                    return json.loads(chat_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON chat history: {e}")
                    return []
            elif isinstance(chat_data, list):
                return chat_data
            else:
                print(f"Unexpected chat_history format: {type(chat_data)}")
                return []
        else:
            print(f"No chat history found for session: {session_id}")
            return []
            
    except Exception as e:
        print(f"Error retrieving chat history for session {session_id}: {e}")
        return []


def save_chat_history(session_id, title, history):
    """Save chat history to DynamoDB"""
    try:
        timestamp = int(datetime.datetime.now().timestamp())
        
        # Ensure history is properly formatted
        if not isinstance(history, list):
            print(f"Warning: history is not a list, converting from {type(history)}")
            history = []
        
        # Convert history to JSON string for storage
        history_json = json.dumps(history)
        
        print(f"Saving chat history for session: {session_id}")
        print(f"Title: {title}")
        print(f"History length: {len(history)} messages")
        print(f"Timestamp: {timestamp}")
        
        # Try to convert session_id to int if it's a string of numbers
        try:
            session_id_key = int(session_id) if isinstance(session_id, str) and session_id.isdigit() else session_id
        except:
            session_id_key = session_id
        
        item = {
            'session_id': session_id_key,
            'title': title,
            'last_updated': timestamp,
            'chat_history': history_json
        }
        
        table.put_item(Item=item)
        print(f"Successfully saved chat history for session: {session_id}")
        return True
        
    except Exception as e:
        print(f"Error saving chat history for session {session_id}: {str(e)}")
        # Removed st.error to hide red error box - just log to console
        return False


def delete_chat_history(session_id):
    """Delete chat history for a specific session"""
    try:
        print(f"Deleting chat history for session: {session_id}")
        
        # Handle session_id type conversion for consistency
        try:
            session_id_key = int(session_id) if isinstance(session_id, str) and session_id.isdigit() else session_id
        except:
            session_id_key = session_id
            
        table.delete_item(Key={'session_id': session_id_key})
        print(f"Successfully deleted chat history for session: {session_id}")
        return True
    except Exception as e:
        print(f"Error deleting chat history for session {session_id}: {str(e)}")
        # Removed st.error to hide red error box
        return False


def rename_chat_history(session_id, new_title):
    """Rename a chat history"""
    try:
        timestamp = int(datetime.datetime.now().timestamp())
        print(f"Renaming chat history for session {session_id} to: {new_title}")
        
        # Handle session_id type conversion for consistency
        try:
            session_id_key = int(session_id) if isinstance(session_id, str) and session_id.isdigit() else session_id
        except:
            session_id_key = session_id
        
        table.update_item(
            Key={'session_id': session_id_key},
            UpdateExpression="SET title = :new_title, last_updated = :timestamp",
            ExpressionAttributeValues={
                ':new_title': new_title,
                ':timestamp': timestamp
            }
        )
        print(f"Successfully renamed chat history for session: {session_id}")
        return True
        
    except Exception as e:
        print(f"Error renaming chat history for session {session_id}: {str(e)}")
        # Removed st.error to hide red error box
        return False


# Email configuration
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "").strip()
SENDER_PASSWORD = "vlxoatpdbmiuxsyk"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def is_valid_email(email):
    return EMAIL_REGEX.match(email.strip())


def share_chat_history(session_id, recipient_emails):
    """Share chat history via email"""
    subject = "Test Genie Chat History"

    try:
        response = table.query(KeyConditionExpression=Key('session_id').eq(session_id))
        items = response.get('Items', [])
        if not items or not items[0].get('chat_history'):
            return False, "No chat history found for this session."

        chat = items[0]
        try:
            chat_data = chat['chat_history']
            if isinstance(chat_data, str):
                history = json.loads(chat_data)
            else:
                history = chat_data
        except json.JSONDecodeError as e:
            return False, f"Invalid chat history format: {e}"

        message = f"Please find the Chat History titled '{chat.get('title', 'Untitled')}' below:\n\n"
        for msg in history:
            message += f"{msg}\n\n"

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_emails, msg.as_string())

        return True, None

    except Exception as e:
        return False, str(e)


def convert_decimal(obj):
    """Convert Decimal objects to int/float for JSON serialization"""
    if isinstance(obj, list):
        return [convert_decimal(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    return obj


def download_chat_history_json(session_id):
    """Download chat history as JSON"""
    try:
        response = table.query(KeyConditionExpression=Key('session_id').eq(session_id))
        items = response.get('Items', [])

        if not items:
            return None, None

        chat = items[0]
        title = chat.get('title', 'Untitled Chat')

        chat_clean = convert_decimal(chat)
        chat_history = chat_clean.get('chat_history', '[]')

        safe_title = re.sub(r'[^\w\s-]', '', title).replace(' ', '_')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.json"

        download_data = {
            'session_id': chat_clean['session_id'],
            'title': chat_clean['title'],
            'last_updated': chat_clean['last_updated'],
            'chat_history': chat_history
        }

        json_data = json.dumps(download_data, indent=2)
        return json_data, filename

    except Exception as e:
        print(f"Error downloading chat history: {str(e)}")
        return None, None


def download_chat_history_markdown(session_id):
    """Download chat history as Markdown"""
    try:
        history = get_chat_history(session_id)

        response = table.query(KeyConditionExpression=Key('session_id').eq(session_id))
        items = response.get('Items', [])
        if not items:
            return None
        chat = items[0]
        chat_title = chat.get('title', 'Untitled Chat')

        if not history:
            return None

        markdown_content = f"# Chat History for Session {session_id}\n\n"
        markdown_content += f"## Chat Title: {chat_title}\n\n"
        for msg in history:
            markdown_content += f"### Message\n{msg}\n\n"

        return markdown_content.encode('utf-8')
        
    except Exception as e:
        print(f"Error creating markdown download: {str(e)}")
        return None


def test_dynamodb_connection():
    """Test DynamoDB connection and permissions"""
    try:
        print("Testing DynamoDB connection...")
        
        # Test table access
        table.load()
        print(f"✓ Successfully connected to table: {table.table_name}")
        
        # Test scan operation
        response = table.scan(Limit=1)
        print(f"✓ Scan operation successful, found {response['Count']} items")
        
        # Test basic put operation
        test_item = {
            'session_id': 'test_connection_' + str(datetime.datetime.now().timestamp()),
            'title': 'Test Connection',
            'last_updated': int(datetime.datetime.now().timestamp()),
            'chat_history': json.dumps(['Test message'])
        }
        table.put_item(Item=test_item)
        print("✓ Put operation successful")
        
        # Clean up test item
        table.delete_item(Key={'session_id': test_item['session_id']})
        print("✓ Delete operation successful")
        
        print("✓ All DynamoDB operations working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ DynamoDB connection test failed: {str(e)}")
        return False


def main():
    """Test SMTP login and DynamoDB connection"""
    print("=== Testing Connections ===")
    
    # Test DynamoDB
    print("\n1. Testing DynamoDB...")
    test_dynamodb_connection()
    
    # Test SMTP
    print("\n2. Testing SMTP...")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = os.environ.get("SENDER_EMAIL", "")
    smtp_password = os.environ.get("SENDER_EMAIL_PASSWORD", "")

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        print("✓ SMTP login successful")
        server.quit()
    except Exception as e:
        print(f"✗ SMTP login failed: {e}")


if __name__ == "__main__":
    main()