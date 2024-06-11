# from pytube import YouTube

# # 다운로드할 유튜브 비디오의 URL
# youtube_url = "https://www.youtube.com/watch?v=0zRLLqpvS3Q"

# # YouTube 객체 생성
# yt = YouTube(youtube_url)

# # 가장 높은 화질의 비디오 스트림 선택
# stream = yt.streams.get_highest_resolution()

# # 파일 경로와 함께 파일명 설정하여 비디오 다운로드
# download_path = "c:/yolo/"  # 저장할 디렉토리 경로 설정
# file_name = "my_video.mp4"  # 저장할 파일명 설정
# stream.download(output_path=download_path, filename=file_name)


import requests
from bs4 import BeautifulSoup
import datetime
import time

# 파일 저장을 위한 경로 설정 및 쓰기 모드로 파일 열기
output_file_path = 'c:/study/list.csv'
with open(output_file_path, 'w', encoding='utf-8-sig') as file:
    file.write("date,회사명,직무,URL,지원방식\n")

    # 검색 키워드 설정
    keyword = 'AI'

    # 1부터 4까지의 페이지를 크롤링
    for page_num in range(1, 5):
        # 사람인 검색 URL
        url = f"https://www.saramin.co.kr/zf_user/search/recruit?search_area=main&search_done=y&search_optional_item=n&searchType=search&searchword={keyword}&recruitPage={page_num}&recruitSort=relation&recruitPageCount=100"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        html = BeautifulSoup(response.text, "html.parser")
        results = html.select("div.item_recruit")

        for ar in results:
            # 회사명, 직무명, URL 추출
            title = ar.select_one("a")['title']
            link = "https://www.saramin.co.kr" + ar.select_one("a")['href']
            company_name = ar.select_one('div.area_corp > strong > a').text.strip()

            # 콤마 제거
            title = title.replace(",", "")
            company_name = company_name.replace(",", "")

            # 현재 날짜
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # 지원 방식 추출, 예외 처리
            try:
                apply_method = ar.select_one("button.sri_btn_xs").text
            except AttributeError:
                apply_method = "홈페이지지원"

            # 파일에 데이터 작성
            file.write(f"{current_date},{company_name},{title},{link},{apply_method}\n")
        
        time.sleep(1)  # 서버 부하 방지를 위한 대기 시간
        print(f"{page_num} 페이지의 {keyword} 채용공고 크롤링을 완료했습니다.")

    print("최종 엑셀 작업 마무리중입니다.")

print("사람인 최종 크롤링이 완료되었습니다.") 