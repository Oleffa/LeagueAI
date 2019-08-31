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
    # TODO find_best()
    # TODO create_experience()
    # TODO path_integration()
    # TODO plot should leave red trail that fades out
    def __init__(self, pc_x_size, pc_y_size, pc_w_e_dim, pc_w_i_dim, vt_active_decay,
                 pc_vt_inject_energy, pc_vt_restore, pc_w_e_var, pc_w_i_var,
                 pc_global_inhib, init_x, init_y, scale_factor):
        self.last_update = time.time()
        self.x_size = pc_x_size
        self.y_size = pc_y_size
        if pc_w_e_dim % 2 == 0:
            self.pc_w_e_dim = pc_w_e_dim + 1
            print("pc_w_e_dim has to be uneven, adding + 1")
        else:
            self.pc_w_e_dim = pc_w_e_dim
        if pc_w_i_dim % 2 == 0:
            self.pc_w_i_dim = pc_w_i_dim + 1
            print("pc_w_i_dim has to be uneven, adding + 1")
        else:
            self.pc_w_i_dim = pc_w_i_dim
        self.posecells = np.zeros((self.x_size, self.y_size))
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        # List to contain all the posecell visual templates (pcvt)
        self.visual_templates = []
        self.current_vt = None
        self.prev_vt = None
        self.best_x = init_x
        self.best_y = init_y
        self.vt_active_decay = vt_active_decay
        self.pc_vt_inject_energy = pc_vt_inject_energy
        self.pc_vt_restore = pc_vt_restore
        self.vt_update = False
        self.pc_w_excite = []
        self.pc_w_inhib = []
        self.pc_w_e_var = pc_w_e_var
        self.pc_w_i_var = pc_w_i_var
        self.pc_global_inhib = pc_global_inhib
        self.scale_factor = scale_factor # This factor increases the np array for visualization only
        self.pc_c_size = 0.5 # How large one cell of the network is

        self.init_pc()
    def init_pc(self):
        self.posecells[self.best_y][self.best_x] = 1.0
        # Set up the gaussians for exciting and inhibiting energies
        # Exciting
        center = int(self.pc_w_e_dim / 2)
        for i in range(0, self.pc_w_e_dim):
            for j in range(0, self.pc_w_e_dim):
                self.pc_w_excite.append(self.norm2d(self.pc_w_e_var, i, j, center))
        # Normalize
        total = sum(self.pc_w_excite)
        self.pc_w_excite = np.true_divide(self.pc_w_excite, total)

        # Inhibtion
        center = int(self.pc_w_i_dim / 2)
        for i in range(0, self.pc_w_i_dim):
            for j in range(0, self.pc_w_i_dim):
                self.pc_w_inhib.append(self.norm2d(self.pc_w_i_var, i, j, center))
        # Normalize
        total = sum(self.pc_w_inhib)
        self.pc_w_inhib = np.true_divide(self.pc_w_inhib, total)

    def update(self):
        self.last_update = time.time()
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.excite()
        self.posecells = self.posecells_new.copy()
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.inhibit()
        self.global_inhib()
        self.normalize()

    def path_integration(self, vtrans, angle):
        time_diff = time.time() - self.last_update
        self.last_update = time.time()
        vtrans = -(vtrans * time_diff) / self.pc_c_size

        temp = int(np.floor(angle * 2 / np.pi))
        pca90 = np.rot90(self.posecells, temp)
        angle = angle - temp

        pca_new = np.zeros([self.x_size + 2, self.y_size + 2])
        pca_new[1:-1, 1:-1] = pca90

        weight_sw = np.power(vtrans, 2) * np.cos(angle) * np.sin(angle)
        weight_se = vtrans * np.sin(angle) * (1.0 - vtrans * np.cos(angle))
        weight_nw = vtrans * np.cos(angle) * (1.0 - vtrans * np.sin(angle))
        weight_ne = 1.0 - weight_sw - weight_se - weight_nw

        #print(weight_sw, weight_se, weight_nw, weight_ne)
        #print(np.round(self.posecells, 2))
        pca_new = pca_new * weight_ne + np.roll(pca_new, 1, 1) * weight_nw + \
                             np.roll(pca_new, 1, 0) * weight_se + \
                             np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

        pca90 = pca_new[1:-1, 1:-1]
        pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
        pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
        pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

        # Rotate back
        self.posecells = np.rot90(pca90, 4 - temp)

        """
        #print(self.posecells)
        weight_matrix_sw = np.full((np.shape(self.posecells)[0] + 1, np.shape(self.posecells)[1] + 1), weight_sw)
        weight_matrix_se = np.full((np.shape(self.posecells)[0] + 1, np.shape(self.posecells)[1] + 1), weight_se)
        weight_matrix_nw = np.full((np.shape(self.posecells)[0] + 1, np.shape(self.posecells)[1] + 1), weight_nw)
        weight_matrix_ne = np.full((np.shape(self.posecells)[0] + 1, np.shape(self.posecells)[1] + 1), weight_ne)
        temp_matrix_sw = np.roll(np.roll(np.multiply(self.posecells, weight_sw), -1, axis=1), 1, axis=0)
        #print(temp_matrix_sw)
        temp_matrix_se = np.roll(np.roll(np.multiply(self.posecells, weight_sw), 1, axis=1), 1, axis=0)
        temp_matrix_nw = np.roll(np.roll(np.multiply(self.posecells, weight_sw), -1, axis=1), -1, axis=0)
        #print(temp_matrix_nw)
        temp_matrix_ne = np.roll(np.roll(np.multiply(self.posecells, weight_sw), 1, axis=1), -1, axis=0)
        self.posecells = temp_matrix_sw + temp_matrix_se + temp_matrix_nw + temp_matrix_ne
        #print(self.posecells)
        """


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
        toprint = cv2.resize(self.posecells, (int(self.x_size*self.scale_factor), int(self.y_size*self.scale_factor)),
                             interpolation=cv2.INTER_AREA)
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
                xw = self.wrap_index(i - int(self.pc_w_e_dim/2), self.x_size)
                yw = self.wrap_index(j - int(self.pc_w_e_dim/2), self.y_size)
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
                xw = self.wrap_index(i - int(self.pc_w_i_dim/2), self.x_size)
                yw = self.wrap_index(j - int(self.pc_w_i_dim/2), self.y_size)
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
        return 1.0 / (std * math.sqrt(2.0 * math.pi)) * math.exp((- math.pow(x - center, 2)
                                                                 - math.pow(y - center, 2))
                                                                 / (2.0 * std * std))


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

class VisualOdometry:
    def __init__(self, frame, scale, draw=True):
        # Parameters for lucas kanade optical flow
        self.size = (int(np.shape(frame)[1]/scale), int(np.shape(frame)[0]/scale))
        frame = cv2.resize(frame, self.size)
        self.prev = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(frame)
        self.hsv[...,1] = 255
        self.draw = draw

    def get_optical_flow(self, now):
        now = cv2.resize(now, self.size)
        now_gray = cv2.cvtColor(now, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev, now_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        angle = np.mean(ang)
        speed = np.mean(mag) * 1
        if self.draw:
            self.hsv[...,0] = ang * 180 / np.pi / 2
            self.hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            x1 = int(np.shape(now)[1]/2)
            y1 = int(np.shape(now)[0]/2)
            x2 = int(round(x1 - speed * 40 * np.cos(angle)))
            y2 = int(round(y1 - speed * 40 * np.sin(angle)))
            bgr = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
            bgr = cv2.line(bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.imshow('frame', bgr)
            cv2.waitKey(30) & 0xff
        self.prev = now_gray
        return speed, angle


