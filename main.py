from tkinter import *
from robert import get_response, bot_name

BG_PANTONE = "#E8E5C9"
BG_COLOR = "aqua"
TEXT_COLOR = "black"

FONT = "Bergen"
FONT_BOLD = "Bergen_bold"

class ChatApp:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        
    def run(self):
        self.window.mainloop()
        
    def _setup_main_window(self):
        self.window.title("NLP Project Group F ")
        self.window.geometry("1280x720")
        self.window.configure(bg=BG_PANTONE)
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text="Chat with Robert", font=FONT_BOLD, pady=10)
        head_label.pack(side=TOP, fill=X)

        # conversation frame
        conversation_frame = Frame(self.window, bg=BG_PANTONE)
        conversation_frame.pack(side=TOP, fill=BOTH, expand=True)

        # text widget
        self.text_widget = Text(conversation_frame, bg=BG_PANTONE, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(conversation_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        scrollbar.configure(command=self.text_widget.yview)

        # input frame
        input_frame = Frame(self.window, bg=BG_COLOR)
        input_frame.pack(side=BOTTOM, fill=X)

        # message entry box
        self.msg_entry = Entry(input_frame, bg=BG_PANTONE, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.pack(side=LEFT, fill=X, expand=True, padx=5, pady=5)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(input_frame, text="Send", font=FONT_BOLD, width=20, bg=BG_PANTONE, command=lambda: self._on_enter_pressed(None))
        send_button.pack(side=RIGHT, padx=5)
     
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You ")
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)
             
        
if __name__ == "__main__":
    app = ChatApp()
    app.run()