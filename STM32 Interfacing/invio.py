#!/usr/bin/env python3
import serial, time, pathlib, sys

PORT = "/dev/ttyACM0"
BAUD = 115200
CSV  = "data19.csv"

ACK_TOKEN = "OK: riga accettata"

def wait_for_ack(ser, timeout=2.0):
    """Legge dalla seriale fino a quando trova l'ACK o scade il timeout"""
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        chunk = ser.read(1024).decode(errors="ignore")
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            buf += chunk
            if ACK_TOKEN in buf:
                return True
        else:
            time.sleep(0.01)
    return False

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1, write_timeout=1)
    except Exception as e:
        print(f"[ERR] Open serial: {e}")
        sys.exit(1)

    time.sleep(0.5)
    print(f"[OK] Porta aperta: {PORT} @ {BAUD}")

    # banner iniziale
    wait_for_ack(ser, timeout=1.0)

    # carica righe
    try:
        lines = [ln.strip() for ln in pathlib.Path(CSV).read_text().splitlines() if ln.strip()]
    except Exception as e:
        print(f"[ERR] Lettura {CSV}: {e}")
        ser.close()
        sys.exit(1)

    print(f"[*] Invierò {len(lines)} righe dal file {CSV}")

    for i, ln in enumerate(lines, 1):
        data = (ln + "\r\n").encode("ascii")
        ser.write(data)
        ser.flush()
        print(f"[TX] riga {i} ({len(data)} byte)")

        if not wait_for_ack(ser, timeout=2.0):
            print(f"[WARN] Nessun ACK per la riga {i}")
            break

    print("[*] Attendo output finale...")
    wait_for_ack(ser, timeout=5.0)

    ser.close()
    print("[OK] Done.")

if __name__ == "__main__":
    main()

