import csv
import time
import re
import isodate
import nltk
import pandas as pd
import numpy as np
import smtplib
from email.message import EmailMessage
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import NoTranscriptFound, NoTranscriptAvailable
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

API_KEYS = [
    "AIzaSyAZs1gGcTtn39lycJQiQA_m1aqgMtwVj4I",
    "AIzaSyD-602gYXckf0F85fbux5uA1eGSLljI43s",
    "AIzaSyD1YIvpHd0N_uo2nIuf3FuOE4l4siA6dcE",
    "AIzaSyCs2Ph6wjqTL1165nHxm1R-fQcjgebOM2Y",
    "AIzaSyCG4R_caE4VKOTrJ-g70UIK0obCJWji5Y4"
]

current_key_index = 0

def get_youtube_service():
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return build("youtube", "v3", developerKey=key)

youtube = get_youtube_service()

def fetch_and_save_videos(genre, filename="output.csv", total_results=500, batch_size=50):
    global youtube
    fieldnames = [
        "Video URL", "Title", "Description", "Channel Title",
        "Keyword Tags", "Category", "Published At", "Duration (seconds)",
        "View Count", "Like Count", "Comment Count", "Subscriber Count",
        "Captions Available", "Caption Text", "Top Comments", "Comment Sentiment Score"
    ]
    with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        video_ids = []
        total_fetched = 0
        next_page_token = None

        while total_fetched < total_results:
            try:
                search_request = youtube.search().list(
                    q=genre,
                    part="id",
                    maxResults=batch_size,
                    type="video",
                    pageToken=next_page_token
                )
                search_response = search_request.execute()
            except Exception as e:
                if "quotaExceeded" in str(e):
                    print("Quota exceeded. Switching API key...")
                    youtube = get_youtube_service()
                    time.sleep(2)
                    continue
                else:
                    print(f"Error during search: {e}")
                    break

            for item in search_response.get("items", []):
                video_ids.append(item["id"]["videoId"])
                total_fetched += 1
                if total_fetched >= total_results:
                    break
            next_page_token = search_response.get("nextPageToken")
            if not next_page_token:
                break

        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            process_video_batch(batch_ids, writer)

    print(f"Data saved to '{filename}'.")
    calculate_final_scores(filename, "video_score.csv")

def process_video_batch(video_ids, writer):
    global youtube
    try:
        video_request = youtube.videos().list(
            id=",".join(video_ids),
            part="snippet,statistics,contentDetails"
        )
        video_response = video_request.execute()

        for video in video_response.get("items", []):
            try:
                video_data = process_single_video(video)
                if video_data:
                    writer.writerow(video_data)
            except Exception as e:
                print(f"Error processing video {video['id']}: {e}")

    except Exception as e:
        if "quotaExceeded" in str(e):
            print("Quota exceeded. Switching API key...")
            youtube = get_youtube_service()
            time.sleep(2)
        else:
            print(f"Error fetching batch data: {e}")

def fetch_subscriber_count(channel_id):
    try:
        request = youtube.channels().list(part="statistics", id=channel_id)
        response = request.execute()
        return int(response["items"][0]["statistics"].get("subscriberCount", 0))
    except Exception as e:
        print(f"Error fetching subscriber count for channel {channel_id}: {e}")
        return 0

def process_single_video(video):
    video_id = video["id"]
    snippet = video["snippet"]
    stats = video["statistics"]
    content = video["contentDetails"]

    captions_available = check_captions(video_id)
    caption_text = fetch_transcript(video_id) if captions_available else "No captions available"
    top_comments = fetch_top_comments(video_id)
    sentiment_score = analyze_comment_sentiment(top_comments)
    channel_id = snippet["channelId"]
    subscriber_count = fetch_subscriber_count(channel_id)

    return {
        "Video URL": f"https://www.youtube.com/watch?v={video_id}",
        "Title": clean_text(snippet["title"]),
        "Description": clean_text(snippet["description"]),
        "Channel Title": clean_text(snippet["channelTitle"]),
        "Keyword Tags": clean_text(", ".join(snippet.get("tags", []))),
        "Category": snippet.get("categoryId", "Unknown"),
        "Published At": snippet["publishedAt"],
        "Duration (seconds)": isodate.parse_duration(content["duration"]).total_seconds(),
        "View Count": int(stats.get("viewCount", 0)),
        "Like Count": int(stats.get("likeCount", 0)),
        "Comment Count": int(stats.get("commentCount", 0)),
        "Subscriber Count": subscriber_count,
        "Captions Available": captions_available,
        "Caption Text": clean_text(caption_text),
        "Top Comments": top_comments,
        "Comment Sentiment Score": sentiment_score
    }

def check_captions(video_id):
    try:
        caption_request = youtube.captions().list(part="id", videoId=video_id)
        caption_response = caption_request.execute()
        return bool(caption_response["items"])
    except Exception:
        return False

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([entry["text"] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript not available"
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return "Error fetching transcript"

def fetch_top_comments(video_id, max_comments=20):
    try:
        comments = []
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText",
            order="relevance"
        )
        comment_response = comment_request.execute()
        for item in comment_response["items"]:
            top_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(clean_text(top_comment))
        return " ||| ".join(comments)
    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {e}")
        return "No comments"

def analyze_comment_sentiment(comment_text):
    if comment_text.strip() == "" or comment_text == "No comments":
        return 0.0
    score = sia.polarity_scores(comment_text)["compound"]
    return round(score, 4)

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_final_scores(input_file, output_file):
    df = pd.read_csv(input_file)

    df["Log_View_Count"] = np.log1p(df["View Count"])
    df["Log_Like_Count"] = np.log1p(df["Like Count"])
    df["Log_Subscriber_Count"] = np.log1p(df["Subscriber Count"])

    for col in ["Log_View_Count", "Log_Like_Count", "Log_Subscriber_Count"]:
        df[col] = 10 * (df[col] / df[col].max())

    df["Adjusted_Sentiment"] = (df["Comment Sentiment Score"] + 1) * 5
    df["Captions_Score"] = df["Captions Available"].apply(lambda x: 1 if str(x).lower() in ["true", "yes", "1"] else 0)

    df["Final Score"] = (
        0.3 * df["Log_Like_Count"] +
        0.3 * df["Log_View_Count"] +
        0.2 * df["Log_Subscriber_Count"] +
        0.15 * df["Adjusted_Sentiment"] +
        0.05 * df["Captions_Score"]
    )

    df["Final Score"] = df["Final Score"].round(2)

    video_counts = df["Channel Title"].value_counts()
    df["Video Count"] = df["Channel Title"].map(video_counts)

    max_videos = df["Video Count"].max()
    df["Weighted Channel Score"] = df.apply(
        lambda row: round(row["Final Score"] + min((row["Video Count"] / max_videos) * 0.5, 0.5), 2),
        axis=1
    )

    df_sorted = df.sort_values(by="Weighted Channel Score", ascending=False)

    # Save only selected columns to video_score.csv (excluding "Weighted Channel Score")
    df_sorted[["Video URL", "Title", "Channel Title", "Final Score"]].to_csv(output_file, index=False)
    print(f"Enhanced scored data saved to '{output_file}'.")

    # Pass full DataFrame to calculate_channel_scores
    calculate_channel_scores(df_sorted, "channel_scores.csv")


def calculate_channel_scores(df, channel_score_file):
    channel_df = df.groupby("Channel Title").agg(
        Average_Final_Score=("Final Score", "mean"),
        Number_of_Videos=("Video URL", "count"),
        Average_Weighted_Score=("Weighted Channel Score", "mean")
    ).reset_index()

    channel_df["Average_Final_Score"] = channel_df["Average_Final_Score"].round(2)
    channel_df["Average_Weighted_Score"] = channel_df["Average_Weighted_Score"].round(2)

    channel_df = channel_df.sort_values(by="Average_Weighted_Score", ascending=False)

    channel_df.to_csv(channel_score_file, index=False)
    print(f"Channel summary saved to '{channel_score_file}'.")


def send_email_report(sender_email, app_password, receiver_email,query):
    try:
        msg = EmailMessage()
        msg["Subject"] = "YouTube Score Reports for "+query
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(
            "Hello,\n\nPlease find attached the YouTube scoring reports:\n- video_score.csv\n- channel_scores.csv\n- output.csv\nBest Regards,\nYouTube Scraper Bot"
        )

        files_to_attach = ["video_score.csv", "channel_scores.csv","output.csv"]

        for filename in files_to_attach:
            try:
                with open(filename, "rb") as f:
                    data = f.read()
                    msg.add_attachment(
                        data,
                        maintype="application",
                        subtype="octet-stream",
                        filename=filename
                    )
                    print(f"✔️ Attached: {filename}")
            except FileNotFoundError:
                print(f"⚠️ Warning: {filename} not found. Skipping attachment.")
            except Exception as e:
                print(f"❌ Error attaching {filename}: {e}")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)
            print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    query = input("Enter the query: ")
    mail=input("enter your mail id: ")
    fetch_and_save_videos(query, total_results=10)

    # Send Email after processing
    send_email_report(
        sender_email="meetsai2004@gmail.com",
        app_password="qptn bmfm pgjp gpjw",  # Your 16-character App Password
        receiver_email=mail,
        query=query
    )
