import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import os


class MaterialPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Material Sound Predictor")
        self.root.geometry("820x560")
        self.root.minsize(760, 520)

        # project root folder
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.predict_file_path = tk.StringVar(value="No MP3 file selected")

        # ---------- Colors ---------- used vscode to pick hex
        self.bg_color = "#0f172a"         
        self.card_color = "#1e293b"       
        self.accent_color = "#38bdf8"     
        self.button_color = "#2563eb"     
        self.button_hover = "#1d4ed8"
        self.text_color = "#f8fafc"       
        self.subtle_text = "#cbd5e1"
        self.output_bg = "#020617"

        # ---------- Root styling ----------
        self.root.configure(bg=self.bg_color)

        # ---------- Header ----------
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill="x", pady=(18, 8))

        title = tk.Label(
            header_frame,
            text="Material Identification Through Sound",
            font=("Segoe UI", 20, "bold"),
            fg=self.text_color,
            bg=self.bg_color
        )
        title.pack()

        subtitle = tk.Label(
            header_frame,
            text="Dataset management, model training, and MP3 prediction",
            font=("Segoe UI", 10),
            fg=self.subtle_text,
            bg=self.bg_color
        )
        subtitle.pack(pady=(4, 0))

        # ---------- Main card ----------
        main_card = tk.Frame(
            self.root,
            bg=self.card_color,
            bd=0,
            highlightthickness=1,
            highlightbackground="#334155"
        )
        main_card.pack(fill="both", expand=True, padx=20, pady=15)

        # ---------- File selection section ----------
        file_section = tk.Frame(main_card, bg=self.card_color)
        file_section.pack(fill="x", padx=20, pady=(20, 10))

        file_label = tk.Label(
            file_section,
            text="Selected MP3 File",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_color,
            bg=self.card_color
        )
        file_label.pack(anchor="w")

        entry_frame = tk.Frame(file_section, bg=self.card_color)
        entry_frame.pack(fill="x", pady=(8, 0))

        self.file_entry = tk.Entry(
            entry_frame,
            textvariable=self.predict_file_path,
            font=("Segoe UI", 10),
            bd=0,
            relief="flat",
            bg="#e2e8f0",
            fg="#0f172a",
            insertbackground="#0f172a"
        )
        self.file_entry.pack(side="left", fill="x", expand=True, ipady=10, padx=(0, 10))

        browse_button = tk.Button(
            entry_frame,
            text="Browse MP3",
            font=("Segoe UI", 10, "bold"),
            bg=self.button_color,
            fg="white",
            activebackground=self.button_hover,
            activeforeground="white",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
            command=self.browse_file
        )
        browse_button.pack(side="right")

        # ---------- Button section ----------
        button_section = tk.Frame(main_card, bg=self.card_color)
        button_section.pack(fill="x", padx=20, pady=12)

        build_button = tk.Button(
            button_section,
            text="Refresh Dataset",
            font=("Segoe UI", 10, "bold"),
            bg="blue",
            fg="white",
            activebackground="light blue",
            activeforeground="white",
            bd=0,
            padx=50,
            pady=12,
            width=16,
            cursor="hand2",
            command=self.build_dataset
        )
        build_button.grid(row=0, column=0, padx=6, pady=6)

        predict_button = tk.Button(
            button_section,
            text="Predict",
            font=("Segoe UI", 10, "bold"),
            bg="orange",
            fg="white",
            activebackground="chocolate1",
            activeforeground="white",
            bd=0,
            padx=50,
            pady=12,
            width=16,
            cursor="hand2",
            command=self.predict_audio
        )
        predict_button.grid(row=0, column=2, padx=6, pady=6)

        clear_button = tk.Button(
            button_section,
            text="Clear Output",
            font=("Segoe UI", 10, "bold"),
            bg="grey",
            fg="white",
            activebackground="azure2",
            activeforeground="white",
            bd=0,
            padx=50,
            pady=12,
            width=16,
            cursor="hand2",
            command=self.clear_output
        )
        clear_button.grid(row=0, column=3, padx=6, pady=6)

        # ---------- Output label ----------
        output_label = tk.Label(
            main_card,
            text="Console Output",
            font=("Segoe UI", 11, "bold"),
            fg=self.text_color,
            bg=self.card_color
        )
        output_label.pack(anchor="w", padx=20, pady=(8, 6))

        # ---------- Output box ----------
        self.output_box = scrolledtext.ScrolledText(
            main_card,
            height=18,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.output_bg,
            fg="white",
            insertbackground="white",
            bd=0,
            relief="flat"
        )
        self.output_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.write_output("Welcome. Select an MP3 file, then choose an action.\n")

    def write_output(self, text):
        self.output_box.insert(tk.END, text)
        self.output_box.see(tk.END)
        self.root.update_idletasks()

    def clear_output(self):
        self.output_box.delete("1.0", tk.END)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select MP3 file",
            filetypes=[("MP3 Files", "*.mp3")]
        )

        if file_path:
            if not file_path.lower().endswith(".mp3"):
                messagebox.showwarning("Invalid File", "Please select an MP3 file only.")
                return
            self.predict_file_path.set(file_path)

    def run_command(self, command):
        self.write_output("\n" + "=" * 70 + "\n")
        self.write_output("Running command:\n")
        self.write_output(" ".join(command) + "\n\n")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_folder
            )

            if result.stdout:
                self.write_output(result.stdout + "\n")

            if result.stderr:
                self.write_output(result.stderr + "\n")

            if result.returncode == 0:
                self.write_output("Action completed successfully.\n")
            else:
                self.write_output(f"Command failed with code {result.returncode}.\n")

        except Exception as error:
            messagebox.showerror("Error", str(error))

    def build_dataset(self):
        command = [
            "python",
            "src/main.py",
            "--mode", "build",
            "--audio_list", "Data/audio_list.csv",
            "--data_csv", "Data/features_dataset.csv"
        ]
        self.run_command(command)


    def predict_audio(self):
        file_path = self.predict_file_path.get().strip()

        if not file_path or file_path == "No MP3 file selected":
            messagebox.showwarning("Missing File", "Please select an MP3 file first.")
            return

        if not os.path.exists(file_path):
            messagebox.showwarning("File Not Found", "The selected MP3 file does not exist.")
            return

        if not file_path.lower().endswith(".mp3"):
            messagebox.showwarning("Invalid File", "Only MP3 files are allowed.")
            return

        command = [
            "python",
            "src/main.py",
            "--mode", "predict",
            "--model_out", "Data/model.pkl",
            "--predict_file", file_path
        ]
        self.run_command(command)


if __name__ == "__main__":
    root = tk.Tk()
    app = MaterialPredictorGUI(root)
    root.mainloop()