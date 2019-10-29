import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

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
    def __init__(self, pc_x_size, pc_y_size, pc_w_e_dim, pc_w_i_dim,
                 find_best_kernel_size, vt_active_decay,
                 pc_vt_inject_energy, pc_vt_restore, pc_w_e_var, pc_w_i_var,
                 pc_global_inhib, init_x, init_y, scale_factor, pc_cells_average):
        self.last_update = time.time()
        self.x_size = pc_x_size
        self.y_size = pc_y_size
        if self.x_size != self.y_size:
            print("Warning: different xy posecell network dimension are untested!")
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
        self.find_best_kernel_size = find_best_kernel_size
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
        self.pc_c_size = 0.1175 # How large one cell of the network is (helps you scale the velocity to the map)
        self.map_image = cv2.resize(cv2.imread('graphics/minimap.png'), (self.x_size * scale_factor, self.y_size * scale_factor))
        self.path = []
        self.pc_cells_average = pc_cells_average

        self.init_pc()

        self.pc_xy_sum_sin_lookup = []
        self.pc_xy_sum_cos_lookup = []
        self.init_population_vector_decoding()

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

    def init_population_vector_decoding(self):
        if self.x_size != self.y_size:
            print("WARNING: in pouplation vector decoding, pc size x and y are not equal!")
        for i in range(0, self.x_size):
            self.pc_xy_sum_cos_lookup.append(np.cos((i+1) * 2.0 * np.pi / self.x_size))
            self.pc_xy_sum_sin_lookup.append(np.sin((i+1) * 2.0 * np.pi / self.x_size))

    def update(self, vtrans, angle):
        self.last_update = time.time()
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.excite()
        self.posecells = self.posecells_new.copy()
        self.posecells_new = np.zeros((self.x_size, self.y_size))
        self.inhibit()
        self.global_inhib()
        self.normalize()
        self.path_integration(vtrans, angle)
        return self.find_best()

    def path_integration(self, vtrans, angle):
        angle += np.pi
        time_diff = time.time() - self.last_update
        self.last_update = time.time()
        if vtrans < 0:
            vtrans *= -1
            angle += np.pi
        while angle < 0:
            angle += 2.0 * np.pi
        while angle > 2.0 * np.pi:
            angle -= 2.0 * np .pi
        #print(vtrans, angle)
        vtrans = (vtrans * time_diff) / self.pc_c_size

        if angle == 0:
            # Move northwest
            self.posecells = self.posecells * (1.0 - vtrans) + \
                np.roll(self.posecells, 1,1) * vtrans
        if angle == np.pi/2:
            # Move northeast
            self.posecells = self.posecells * (1 - vtrans) + \
                np.roll(self.posecells, 1, 0) * vtrans
        if angle == np.pi:
            # Move southeast
            self.posecells = self.posecells * (1.0 - vtrans) + \
                np.roll(self.posecells, -1, 1) * vtrans
        if angle == 3*np.pi/2:
            # Move southeast
            self.posecells = self.posecells * (1.0 - vtrans) + \
                np.roll(self.posecells, -1, 0) * vtrans
        else:
            temp = np.floor(angle * 2 / np.pi)
            pca90 = np.rot90(self.posecells, temp)
            dir90 = angle - temp * np.pi/2
            pca_new = np.zeros([self.x_size + 2, self.y_size + 2])
            pca_new[1:-1, 1:-1] = pca90

            weight_sw = (vtrans**2) * np.cos(dir90) * np.sin(dir90)
            #weight_se = vtrans * np.sin(dir90) - (vtrans**2) * np.cos(dir90) * np.sin(dir90)
            #weight_nw = vtrans * np.cos(dir90) - (vtrans**2) * np.cos(dir90) * np.sin(dir90)
            weight_se = vtrans * np.sin(dir90) * (1 - vtrans * np.cos(dir90))
            weight_nw = vtrans * np.cos(dir90) * (1 - vtrans * np.sin(dir90))
            weight_ne = 1.0 - weight_sw - weight_se - weight_nw

            #print(weight_sw, weight_se, weight_nw, weight_ne)

            pca_new = pca_new * weight_ne + np.roll(pca_new, 1, 1) * weight_nw + \
                                 np.roll(pca_new, 1, 0) * weight_se + \
                                 np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

            pca90 = pca_new[1:-1, 1:-1]
            pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
            pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
            pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

            # Rotate back
            self.posecells = np.rot90(pca90, 4 - temp)

    def find_best(self):

        # Find max activation cell
        y, x = np.where(self.posecells == np.max(self.posecells))
        x = x[0]
        y = y[0]
        sum_array = self.posecells[y-self.pc_cells_average:y+self.pc_cells_average, x-self.pc_cells_average:x+self.pc_cells_average]
        # We do not need to account for the wrap around here, because you cant wrap around in the league map
        x_sums = [0] * self.x_size
        y_sums = [0] * self.y_size
        x_sums[x-self.pc_cells_average:x+self.pc_cells_average] = np.sum(sum_array, axis=0)
        y_sums[y-self.pc_cells_average:y+self.pc_cells_average] = np.sum(sum_array, axis=1)

        sum_x1 = 0
        sum_x2 = 0
        sum_y1 = 0
        sum_y2 = 0
        for i in range(0, self.x_size):
            sum_x1 += self.pc_xy_sum_sin_lookup[i] * x_sums[i]
            sum_x2 += self.pc_xy_sum_cos_lookup[i] * x_sums[i]
            sum_y1 += self.pc_xy_sum_sin_lookup[i] * y_sums[i]
            sum_y2 += self.pc_xy_sum_cos_lookup[i] * y_sums[i]
        v_x = math.atan2(sum_x1, sum_x2) * self.x_size / (2.0 * np.pi) - 1
        v_y = math.atan2(sum_y1, sum_y2) * self.y_size / (2.0 * np.pi) - 1
        while v_x < 0:
            v_x += self.x_size
        while v_x >= self.x_size:
            v_x -= self.x_size
        while v_y < 0:
            v_y += self.y_size
        while v_y >= self.y_size:
            v_y -= self.y_size

        return v_x, v_y

    def on_view_template(self, vt):
        if (vt >= len(self.visual_templates)):
            # Template does not exist yet
            self.visual_templates.append(PosecellVisualTemplate(len(self.visual_templates), self.best_x,
                                                                 self.best_y, self.vt_active_decay))
        else:
            # Re-use existing template
            pcvt = self.visual_templates[vt]
            # Prevent injection in recently created visual templates
            if vt < len(self.visual_templates) - 5:
                if vt != self.current_vt:
                    pass
                else:
                    pcvt.decay += self.vt_active_decay

                energy = self.pc_vt_inject_energy * 1.0 / 30.0 * (30.0 - math.exp(1.2 * pcvt.decay))
                if energy > 0:
                    print("injecting at : ", vt)
                    self.inject(pcvt.pcvt_x, pcvt.pcvt_y, energy)
        for temp_vt in self.visual_templates:
            temp_vt.decay -= self.pc_vt_restore
            if temp_vt.decay < self.vt_active_decay:
                temp_vt.decay = self.vt_active_decay
        self.prev_vt = self.current_vt
        self.current_vt = vt
        self.vt_update = True

    def plot_posecell_network(self, fps, memory, global_x, global_y):
        toprint = self.map_image.astype(np.float64)
        pcs = np.zeros(toprint.shape, dtype=toprint.dtype)
        pcs[:,:,1] = cv2.resize(self.posecells * 255.0, (int(self.x_size*self.scale_factor),
                                                 int(self.y_size*self.scale_factor)),
                                interpolation=cv2.INTER_AREA)
        toprint = cv2.addWeighted(toprint, 0.75, pcs, 1.0, 0).astype(np.uint8)
        center = (int(global_x * self.scale_factor), int(global_y * self.scale_factor))
        cv2.circle(toprint, center, self.pc_w_e_dim*2, (0, 255, 0), 1)
        cv2.circle(toprint, center, 3, (0, 0, 255), -1)

        a = 0
        if len(self.path) > 0:
            a = np.sqrt(np.power(global_x * self.scale_factor - self.path[-1][0], 2) + np.power(global_y * self.scale_factor - self.path[-1][1], 2))
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                cv2.line(toprint, self.path[i-1], self.path[i], (0, 0 , 255), 1)
            cv2.line(toprint, self.path[-1], center, (0, 0 , 255), 1)
        if len(self.path) == 0 or a > 6.0:
            self.path.append(center)


        cv2.imshow("Posecell Network", toprint)
        cv2.waitKey(int(1000/fps))

    def inject(self, x, y, energy):
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
            #print("Match ID: ", temp_match_id, " / ", len(self.memory)-1, " temp_error: ", temp_error)
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
            cur_error = cv2.absdiff(input_frame, memory_template.visual_data)
            cur_error = np.mean(cur_error)
            if cur_error < smallest_error:
                smallest_error = cur_error
                best_match_id = id

        return best_match_id, smallest_error

class VisualOdometry:
    """
    Notes:
        - There is substential drift in the visual odometry
        - The median of the optical flow is maybe not the best solution, filtering out the non majority values could help
    """
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
        flow = cv2.calcOpticalFlowFarneback(self.prev, now_gray, None, 0.5, 20, 5, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        angle = np.median(ang)
        speed = np.median(mag)
        # TODO this is not really a nice solution but the velocity left right is higher than up down usually. So i clipped it.
        if speed > 0.1:
            speed = 1.0
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


