import subprocess

if __name__ == "__main__":
    ips = []
    for i in range(1, 255):
        ip = "192.168.12." + str(i)
        complete = subprocess.run(["ping -c 2 " + ip], shell=True)
        if complete.returncode == 0:
            ips.append(ip)

    print(ips)