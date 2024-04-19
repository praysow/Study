from pytube import YouTube

# 다운로드할 유튜브 비디오의 URL
youtube_url = "https://www.youtube.com/watch?v=0zRLLqpvS3Q"

# YouTube 객체 생성
yt = YouTube(youtube_url)

# 가장 높은 화질의 비디오 스트림 선택
stream = yt.streams.get_highest_resolution()

# 파일 경로와 함께 파일명 설정하여 비디오 다운로드
download_path = "c:/yolo/"  # 저장할 디렉토리 경로 설정
file_name = "my_video.mp4"  # 저장할 파일명 설정
stream.download(output_path=download_path, filename=file_name)
