@echo off
setlocal enabledelayedexpansion
title Drowsiness Detection System
mode con: cols=80 lines=25
color 0B

:: --- PHAN THONG TIN DEVELOPER ---
echo ============================================================
echo        HE THONG CANH BAO BUON NGU - DROWSINESS DETECTION
echo ============================================================
echo  [+] Developed by: Han Vu
echo  [+] Version     : 1.0.0
echo  [+] Institution : Truong Dai hoc Cong nghiep Ha Noi
echo  [+] Project     : Do an Tot nghiep Khoa hoc May tinh
echo ------------------------------------------------------------
echo.

:: Di chuyển vào thư mục chứa file bat
set SCRIPTPATH=%~dp0
cd /d %SCRIPTPATH%

:: --- HIEU UNG THANH PROGRESS BAR GIA LAP ---
echo [INFO] Dang kiem tra cac thanh phan he thong...
set "progress=                                        "
set "fill=########################################"

for /L %%i in (1,1,40) do (
    set "line=!fill:~0,%%i!!progress:~%%i,40!"
    set /a "percent=%%i * 100 / 40"
    <nul set /p "=Progress: [!line!] !percent!%% "
    timeout /t 0 /nobreak >nul
    <nul set /p "= "
)
echo.
echo [DONE] Moi truong da san sang.
echo.

:: Kiểm tra xem thư mục môi trường ảo 'drowsiness' có tồn tại không
if exist "drowsiness\Scripts\activate.bat" (
    echo [INFO] Dang kich hoat venv...
    call drowsiness\Scripts\activate.bat
) else (
    color 0C
    echo [ERROR] Khong tim thay thu muc moi truong ao 'drowsiness'!
    echo [HELP] Vui long dam bao thu muc 'drowsiness' nam cung cap voi file .bat nay.
    pause
    exit /b
)

:: Chạy chương trình chính
echo [INFO] System is starting...
echo ------------------------------------------------------------
python main.py

:: Giữ cửa sổ cmd nếu chương trình bị tắt hoặc lỗi
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [CRITICAL] He thong dung dot ngot voi loi: %errorlevel%
)
echo ------------------------------------------------------------
echo Cam on ban da su dung he thong!
pause