# import tkinter as tk
# import tkinter.ttk as ttk
# import os


# class UrlshortnerWidget(tk.Toplevel):
#     def __init__(self, master=None, **kw):
#         super(UrlshortnerWidget, self).__init__(master, **kw)
#         self.frm_main = tk.Frame(self)
#         self.lbl_app_name = tk.Label(self.frm_main)
#         self.lbl_app_name.configure(
#             anchor="n",
#             background="#6200c4",
#             font="{Arial} 24 {bold}",
#             foreground="#ffffff",
#         )
#         self.lbl_app_name.configure(
#             takefocus=False, text="Lung X-Ray Classifier")
#         self.lbl_app_name.pack(pady="30", side="top")
#         self.ent_long_url = tk.Entry(self.frm_main)
#         self.long_url = tk.StringVar(value="")
#         self.ent_long_url.configure(
#             font="{Arial} 12 {}",
#             justify="center",
#             state="normal",
#             textvariable=self.long_url,
#         )
#         self.ent_long_url.configure(validate="focusin", width="80")
#         self.ent_long_url.pack(side="top")
#         self.btn_shorten = tk.Button(self.frm_main)
#         self.btn_shorten.configure(
#             background="#ffffff",
#             font="{Arial} 16 {bold}",
#             foreground="#8000ff",
#             text="Click to Shorten",
#         )
#         self.btn_shorten.pack(pady="50", side="top")
#         self.btn_shorten.configure(command=self.make_short_link)
#         self.ent_short_url = tk.Entry(self.frm_main)
#         self.ent_short_url.configure(
#             background="#6200c4",
#             borderwidth="0",
#             font="{Arial} 24 {}",
#             foreground="#ffffff",
#         )
#         self.ent_short_url.configure(
#             justify="center",
#             readonlybackground="#6200c4",
#             relief="flat",
#             state="readonly",
#         )
#         self.ent_short_url.configure(width="25")
#         self.ent_short_url.pack(pady="50", side="top")
#         self.frm_main.configure(background="#6200c4",
#                                 height="480", width="800")
#         self.frm_main.pack(side="top")
#         self.configure(background="#6200c4", height="200", width="200")
#         self.geometry("800x480")
#         self.resizable(False, False)
#         self.title("URL Shortner")

#     def make_short_link(self):
#         os.system('python predict.py')


# if __name__ == "__main__":
#     root = tk.Tk()
#     widget = UrlshortnerWidget(root)
#     # widget.pack(expand=True, fill="both")
#     root.mainloop()

from waitress import serve
import os
from flask import Flask, render_template, request, flash, send_file

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        print(request.form['filepath'])
        print("EEEEE")
    else:
        return render_template('home.html')


serve(
   app.run(),
   host="127.0.0.1",
   port=5000,
   threads=2
)
