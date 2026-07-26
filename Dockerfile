FROM python:3.11-slim

# Thư mục làm việc
WORKDIR /app

# Copy file requirements
COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ dự án
COPY . .

# Mở cổng Streamlit
EXPOSE 8501

# Chạy ứng dụng
CMD ["streamlit", "run", "src/app_mini.py", "--server.address=0.0.0.0", "--server.port=8501"]