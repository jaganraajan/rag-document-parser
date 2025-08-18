class IDStrategy:
    def __init__(self):
        self.current_id = 0

    def generate_id(self):
        self.current_id += 1
        return f"id_{self.current_id}"

    def reset_id(self):
        self.current_id = 0