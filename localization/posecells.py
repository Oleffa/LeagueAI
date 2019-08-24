import numpy as np
import cv2
import time
import math

class PosecellVisualTemplate:
    def __init__(self, i, pcvt_x, pcvt_y, decay):
        self.i = i
        self.pcvt_x = pcvt_x
        self.pcvt_y = pcvt_y
        self.decay = decay

class PosecellExperience:
    def __init__(self, i, pcxp_x, pcxp_y):
        self.i = i
        self.pc_experience_x = pcxp_x
        self.pc_experience_y = pcxp_y

class PosecellNetwork:
    # TODO on_odo function and odometry
    # TODO find_best()
    # TODO create_experience()
    # TODO path_integration()
    # TODO the energy packet is not stable and traveling, possible int rounding errors -> compare to c++ or see the python implementation for 1 d what solves the problem there
    def __init__(self, pc_x_size, pc_y_size, pc_w_e_dim, pc_w_i_dim, vt_active_decay,
                 pc_vt_inject_energy, pc_vt_restore, pc_w_e_var, pc_w_i_var,
                 pc_global_inhib):
        self.x_size = pc_x_size
        self.y_size = pc_y_size
        self.pc_w_e_dim = pc_w_e_dim
        self.pc_w_i_dim = pc_w_i_dim
        self.posecells = np.zeros((self.x_size, self.y_size))
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        # List to contain all the posecell visual templates (pcvt)
        self.visual_templates = []
        self.current_vt = None
        self.prev_vt = None
        self.best_x = int(self.x_size / 2)
        self.best_y = int(self.y_size / 2)
        self.vt_active_decay = vt_active_decay
        self.pc_vt_inject_energy = pc_vt_inject_energy
        self.pc_vt_restore = pc_vt_restore
        self.vt_update = False
        self.pc_w_excite = []
        self.pc_w_inhib = []
        self.pc_w_e_var = pc_w_e_var
        self.pc_w_i_var = pc_w_i_var
        self.pc_global_inhib = pc_global_inhib

        self.init_pc()
    def init_pc(self):
        self.posecells[self.best_y][self.best_x] = 1.0
        # Set up the gaussians for exciting and inhibiting energies
        # Exciting
        center = self.pc_w_e_dim / 2;
        for i in range(0, self.pc_w_e_dim):
            for j in range(0, self.pc_w_e_dim):
                self.pc_w_excite.append(self.norm2d(self.pc_w_e_var, i, j, center))
        # Normalize
        total = sum(self.pc_w_excite)
        for i in self.pc_w_excite:
            i /= total
        # Inhibtion
        for i in range(0, self.pc_w_e_dim):
            for j in range(0, self.pc_w_e_dim):
                self.pc_w_inhib.append(self.norm2d(self.pc_w_i_var, i, j, center))
        # Normalize
        total = sum(self.pc_w_inhib)
        for i in self.pc_w_inhib:
            i /= total
        print(self.pc_w_excite)

    def on_update(self, vtrans_x, vtrans_y):
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.excite()
        self.posecells = self.posecells_new.copy()
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.inhibit()
        self.global_inhib()
        self.normalize()

    def on_view_template(self, vt):
        print("on vt: ", vt, " len: ", len(self.visual_templates))
        if (vt >= len(self.visual_templates)):
            # Template does not exist yet
            self.visual_templates.append(PosecellVisualTemplate(len(self.visual_templates), self.best_x,
                                                                 self.best_y, self.vt_active_decay))
        else:
            # Re-use existing template
            pcvt = self.visual_templates[vt]
            # Prevent injection in recently created visual templates
            if vt < len(self.visual_templates) - 10:
                if vt != self.current_vt:
                    pass
                else:
                    pcvt.decay += self.vt_active_decay

                energy = self.pc_vt_inject_energy * 1.0 / 30.0 * (30.0 - math.exp(1.2 * pcvt.decay))
                if energy > 0:
                    self.inject(pcvt.pcvt_x, pcvt.pcvt_y, energy)
        for temp_vt in self.visual_templates:
            temp_vt.decay -= self.pc_vt_restore
            if temp_vt.decay < self.vt_active_decay:
                temp_vt.decay = self.vt_active_decay
        self.prev_vt = self.current_vt
        self.current_vt = vt
        self.vt_update = True

    def plot_posecell_network(self, fps):
        toprint = cv2.resize(self.posecells, (self.x_size*10, self.y_size*10), interpolation=cv2.INTER_AREA)
        cv2.imshow("Posecell Network", toprint)
        cv2.waitKey(int(1000/fps))

    def inject(self, x, y, energy):
        print("injecting", np.sum(self.posecells))
        if x < self.x_size and x >= 0 and y < self.y_size and y >= 0:
            self.posecells[y][x] += energy
        else:
            print("error in posecells.py, injecting at invalid index")

    def excite(self):
        for i in range(0, self.x_size):
            for j in range(0, self.y_size):
                if self.posecells[j][i] != 0:
                    self.excite_helper(i, j)

    def excite_helper(self, x, y):
        excite_index = 0
        for j in range(y, y + self.pc_w_e_dim):
            for i in range(x, x + self.pc_w_e_dim):
                xw = int(self.wrap_index(i - self.pc_w_e_dim/2, self.x_size))
                yw = int(self.wrap_index(j - self.pc_w_e_dim/2, self.y_size))
                self.posecells_new[yw][xw] += self.posecells[y][x] * self.pc_w_excite[excite_index]
                excite_index += 1

    def inhibit(self):
        for i in range(0, self.x_size):
            for j in range(0, self.y_size):
                if self.posecells[j][i] != 0:
                    self.inhibit_helper(i, j)

    def inhibit_helper(self, x, y):
        inhibit_index = 0
        for j in range(y, y + self.pc_w_i_dim):
            for i in range(x, x + self.pc_w_i_dim):
                xw = int(self.wrap_index(i - self.pc_w_i_dim/2, self.x_size))
                yw = int(self.wrap_index(j - self.pc_w_i_dim/2, self.y_size))
                self.posecells_new[yw][xw] += self.posecells[y][x] * self.pc_w_inhib[inhibit_index]
                inhibit_index += 1

    def global_inhib(self):
        self.posecells = self.posecells - self.posecells_new - self.pc_global_inhib
        self.posecells[self.posecells < 0] = 0

    def normalize(self):
        total = np.sum(self.posecells)
        assert total > 0
        self.posecells /= total

    # Helper functions
    def wrap_index(self, index, size):
        while index < 0:
            index += size
        while index >= size:
            index -= size
        return index

    def norm2d(self, std, x, y, center):
        return 1.0 / (std * math.sqrt(2.0 * math.pi)) * math.exp((-math.pow(x - center, 2)
                                                                  -math.pow(y - center, 2)
                                                                  / (2.0 * std * std)))


class Template:
    def __init__(self, i, visual_data, mean):
        self.i = i
        self.visual_data = visual_data
        self.mean = mean

class ViewTemplates:
    def __init__(self, vt_size_x, vt_size_y, vt_x_min,
                 vt_y_min, vt_x_max, vt_y_max, rate=2,
                 template_match_threshold=1.0):
        self.vt_size_x = vt_size_x
        self.vt_size_y = vt_size_y
        self.vt_x_min = vt_x_min
        self.vt_x_max = vt_x_max
        self.vt_y_min = vt_y_min
        self.vt_y_max = vt_y_max
        self.template_match_threshold = template_match_threshold
        self.memory = []
        self.rate = rate                   # Max times on_image gets called per second
        self.last_time = time.time()
        self.cur_match_id = 0

    def on_image(self, frame):
        # Update the match id and error
        if time.time() > self.last_time + self.rate:
            frame, mean = self.preprocess_image(frame)
            temp_match_id, temp_error = self.compare(frame)
            self.last_time = time.time()
            if temp_error <= self.template_match_threshold:
                if len(self.memory) <= 0:
                    # No templates yet, create new one
                    self.memory.append(Template(0, frame, mean))
                else:
                    # Match detected, set id
                    self.cur_match_id = temp_match_id
            else:
                # Create new template
                self.memory.append(Template(len(self.memory), frame, mean))
        return self.cur_match_id

    def preprocess_image(self, frame):
        # Crop
        w = self.vt_x_max - self.vt_x_min
        h = self.vt_y_max - self.vt_y_min
        frame = frame[self.vt_y_min:self.vt_y_min+h, self.vt_x_min:self.vt_x_min+w]
        # Grayscale Image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Normalize
        frame = cv2.normalize(frame, frame, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
        # Downsample
        frame = cv2.resize(frame, (self.vt_size_x, self.vt_size_y), interpolation=cv2.INTER_AREA)
        # Mean
        mean = np.mean(frame)
        return frame, mean

    def compare(self, input_frame):
        best_match_id = 0
        smallest_error = float("inf")
        if len(self.memory) <= 0:
            return best_match_id, 0.0

        # Compute error
        for id, memory_template in enumerate(self.memory):
            cdiff = 0
            for i in range(0, self.vt_size_x):
                for j in range(0, self.vt_size_y):
                    cdiff += abs(input_frame[j][i] - memory_template.visual_data[j][i])
            cur_error = cdiff / (self.vt_size_y * self.vt_size_x)
            if cur_error < smallest_error:
                smallest_error = cur_error
                best_match_id = id
        return best_match_id, smallest_error
