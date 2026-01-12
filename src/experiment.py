import os

class Experiment:
    def __init__(self, save_dir="experiment2", toggle_hand_side=True):
        self.save_dir = save_dir
        self.PERSON_ID = self.get_next_person_id(save_dir)
        self.toggle_hand_side = toggle_hand_side
        self.hand_side = ["L1", "R1"]
        self.hand_index = 0
    
    def get_next_person_id(self, folder):
        max_id = 0
        for f in os.listdir(folder):
            if f.startswith("person_") and f[7:10].isdigit():
                pid = int(f[7:10])
                max_id = max(max_id, pid)
        return max_id + 1
    
    def get_start_filename(self):
        basename = f"person_{self.PERSON_ID:03d}" + (f"_{self.hand_side[self.hand_index]}" if self.toggle_hand_side else "")
        return os.path.join(self.save_dir, basename)
    
    def update_dir(self):
        if not self.toggle_hand_side:
            self.PERSON_ID += 1
            return
        
        self.hand_index += 1
        if self.hand_index == 2:
            self.hand_index = 0
            self.PERSON_ID += 1