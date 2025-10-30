# Script PowerShell để chạy ứng dụng Fraud Detection
# Sử dụng: .\run.ps1

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Phát hiện Gian lận Thẻ Tín dụng - ML Demo  " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python
Write-Host "Đang kiểm tra Python..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "✓ $pythonVersion đã cài đặt" -ForegroundColor Green
} else {
    Write-Host "✗ Python chưa được cài đặt!" -ForegroundColor Red
    Write-Host "  Vui lòng cài đặt Python từ https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Kiểm tra dependencies
Write-Host "Đang kiểm tra dependencies..." -ForegroundColor Yellow
$requirementsExists = Test-Path "requirements.txt"
if ($requirementsExists) {
    Write-Host "✓ requirements.txt tìm thấy" -ForegroundColor Green
    
    # Hỏi người dùng có muốn cài đặt dependencies không
    $install = Read-Host "Bạn có muốn cài đặt/cập nhật dependencies? (y/n)"
    if ($install -eq "y" -or $install -eq "Y") {
        Write-Host "Đang cài đặt dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        Write-Host "✓ Hoàn thành cài đặt dependencies" -ForegroundColor Green
    }
} else {
    Write-Host "✗ requirements.txt không tìm thấy!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Chạy ứng dụng
Write-Host "Đang khởi động ứng dụng Streamlit..." -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Ứng dụng sẽ mở tại: http://localhost:8501  " -ForegroundColor Cyan
Write-Host "  Nhấn Ctrl+C để dừng ứng dụng              " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
