#!/usr/bin/env python3
import time, os, re, io, shutil, paramiko, pysftp
from flask import Flask, render_template, request, redirect, url_for, jsonify
from base64 import encodebytes
from PIL import Image
from pathlib import Path


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode="r")  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")  # encode as base64
    return encoded_img


home_dir = str(Path.home())
app = Flask(__name__, template_folder="template")
# app.config['UPLOAD_FOLDER'] = "/Users/dusc/segmentation//"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def upload_file():
    # delete contents from static folder

    folder = home_dir + "/msirs/static/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if (
                os.path.isfile(file_path)
                or os.path.islink(file_path)
                and not re.findall("home", file_path)
                and not re.findall("agbv|tud-logo|msirs_logo|tu_fk", file_path)
            ):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    uploaded_file = request.files["file"]
    if uploaded_file.filename != "":
        path = f"static/query{str(uploaded_file.filename)[-4:]}"
        uploaded_file.save(uploaded_file.filename)
        shutil.move(str(uploaded_file.filename), path)
        server = Server()
        server.connection.put(
            path, f"/home/{server.username}/query/" + str(uploaded_file.filename)
        )
        # server.connection.put(path, f"/home/{server.username}/server-test/query{str(uploaded_file.filename)[-4:]}")
        print(uploaded_file.filename)

        # execute pipeline and retrieve results
        # activate venv
        # sleeptime = 0.001
        # outdata, errdata = '', ''
        # ssh_transp = server.ssh.get_transport()
        # chan = ssh_transp.open_session()
        # # chan.settimeout(3 * 60 * 60)
        # chan.setblocking(0)
        # chan.exec_command('source ~/codebase-v1/venv/bin/activate && python3 ~/segmentation/first_full_pipeline.py')
        # while True:  # monitoring process
        #     # Reading from output streams
        #     while chan.recv_ready():
        #         outdata += str(chan.recv(1000))
        #     while chan.recv_stderr_ready():
        #         errdata += str(chan.recv_stderr(1000))
        #     if chan.exit_status_ready():  # If completed
        #         break
        #     time.sleep(sleeptime)
        # retcode = chan.recv_exit_status()
        # ssh_transp.close()

        # print(outdata)
        # print(errdata)

        # TODO: dddddddd
        stdin, stdout, stderr = server.ssh.exec_command(
            "source ~/codebase-v1/venv/bin/activate && python3 ~/msirs/pipeline_v2_2_query.py"
        )
        stdout.channel.recv_exit_status()
        lines = stdout.readlines()
        for line in lines:
            print(line)
        print("Venv activated")
        # execute script
        # stdin, stdout, stderr = server.ssh.exec_command("")
        # output = stdout.readlines()
        # for item in output:
        #     item = item[:-1]
        #     print(item)
        # print("Pipeline finished..")
        stdin, stdout, stderr = server.ssh.exec_command(
            "ls ~/server-test | grep retrieval"
        )
        output = stdout.readlines()
        for item in output:
            item = item[:-1]
            print(item)
            server.connection.get(
                f"/home/{server.username}/server-test/{item}",
                home_dir + f"/msirs/static/{item}",
            )

        # close connections
        server.connection.close()
        server.ssh.close()
        print("Closes connections.")

        return redirect("results")
    else:
        return url_for("home")


@app.route("/results")
def results():
    res_path = home_dir + "/msirs/static/"
    files = os.listdir(res_path)
    # results = [res_path+i for i in files if re.findall("result", i) or re.findall("query", i)]
    results = [i for i in files if re.findall("retrieval", i)]
    print(results)
    return render_template("results.html", binderList=results)


@app.route("/upload_success")
def upload_suc():
    return render_template("success.html")


@app.route("/upload_failed")
def upload_fail():
    return render_template("failed.html")


@app.route("/upload")
def upload_page():
    return render_template("upload_page.html")


@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files["file"]
    if uploaded_file.filename != "":
        print("Received file")
        server = Server()
        # path = f"static/query{str(uploaded_file.filename)[-4:]}"
        # uploaded_file.save(uploaded_file.filename)
        stdin, stdout, stderr = server.ssh.exec_command(
            f"source ~/codebase-v1/venv/bin/activate && python3 ~/msirs/pipeline_v2_2_import.py {uploaded_file.filename}"
        )

    if 0 == 1:
        return redirect("/upload_success")
    else:
        return redirect("/upload_failed")


class Server:
    def __init__(self):
        self.hostname = "jabba.king-little.ts.net"
        self.username = "pg2022"
        password = "isthatthemars42"
        port = 2222
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        try:
            self.connection = pysftp.Connection(
                host=self.hostname,
                username=self.username,
                password=password,
                port=port,
                cnopts=cnopts,
            )
        except Exception as err:
            raise Exception(err)
        finally:
            print(f"Connected to {self.hostname} as {self.username}.")

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            self.hostname, username=self.username, password=password, port=port
        )


if __name__ == "__main__":
    app.run(debug=True)
