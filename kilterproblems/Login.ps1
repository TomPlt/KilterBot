# Login.ps1

$deviceID = "localhost:XXXX"  # Adjust as per your actual device/emulator ID

adb -s $deviceID shell monkey -p com.auroraclimbing.kilterboard 1
Start-Sleep -Seconds 5

# Navigate UI and Enter Credentials
adb -s $deviceID shell input keyevent KEYCODE_TAB
Start-Sleep -Seconds 2
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_ENTER
# Login
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_TAB
# adb -s $deviceID shell input text <username>
adb -s $deviceID shell input keyevent KEYCODE_TAB
# adb -s $deviceID shell input text <password>
adb -s $deviceID shell input keyevent KEYCODE_TAB
adb -s $deviceID shell input keyevent KEYCODE_ENTER

# Extract DB after an appropriate delay post-login
Start-Sleep -Seconds 20
adb -s $deviceID pull /data/data/com.auroraclimbing.kilterboard/files/db-264.sqlite3
