import numpy as np
import pandas as pd
import random
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from matplotlib import pyplot as plt

class Course:
    course_list: list = list()
    courses: pd.DataFrame
    ids: np.ndarray
    n_lessons: np.ndarray
    n_students: np.ndarray
    course_array: np.ndarray
    years_subsets: list[list] = list(list())

    def __init__(
        self,
        id: int,
        n_lessons: int,
        n_students: int,
        teacher = None,
        year: int = None,
    ):
        self.id = int(id)
        self.n_lessons = int(n_lessons)
        self.n_students = int(n_students)
        self.teacher = teacher
        self.teacher.add_course(self)
        self.year = year

        self.blocks = []
        self.blocks = self.get_blocks


    @property
    def get_blocks(self):
        """
        Dynamically calculate the blocks of lectures based on n_lessons.
        """
        if len(self.blocks) == 0:
            n = self.n_lessons
            blocks = []
            while n > 0:
                if n == 4:
                    if random.choice([True, True, False]):
                        blocks.append(4)
                        n -= 4
                    else:
                        blocks.append(min(2, n))
                        n -= blocks[-1]
                elif n % 3 == 0 or n == 5:
                    blocks.append(3)
                    n -= 3
                elif n == 2:
                    blocks.append(2)
                    n -= 2
                else:
                    raise Exception("ERROR")
               

                
            self.blocks = blocks
        return self.blocks
    
    def iter_block_size(self):
        blocks = self.get_blocks
        for block in blocks:
            yield block

    @staticmethod
    def generate_courses(n_courses: int, lesson_count: tuple, student_count: tuple, teachers: list):
        
        Course.ids = np.arange(n_courses)
        Course.n_lessons = np.random.randint(*lesson_count, (n_courses,))
        Course.n_students = np.random.randint(*student_count, (n_courses,))
        Course.course_array = np.column_stack([Course.ids, Course.n_lessons, Course.n_students])
        teacher_course_list = []
        course_year_list = []
        n_years = 4

        for i, course in enumerate(Course.course_array):
            course_per_year = n_courses // n_years
            year = i // course_per_year
            id, n_lessons, n_students = course
            new_course = Course(id, n_lessons, n_students, teacher=random.choice(teachers), year=year)
            if len(Course.years_subsets) < year + 1:
                Course.years_subsets.append([])
            Course.years_subsets[year].append(new_course)
            Course.course_list.append(new_course)
            course_year_list.append(new_course.year)
            teacher_course_list.append(new_course.teacher.id)

        teacher_course_array = np.array(teacher_course_list)
        Course.course_array = np.column_stack([Course.course_array, teacher_course_array,  np.array(course_year_list)])

        Course.courses = pd.DataFrame(Course.course_array, columns=["id", "n_lessons", "n_students", "teacher", "year"], index=None)
        Course.display()

    @staticmethod
    def display():
        #print(f"n_courses: {len(Course.courses)}")
        #print(f"course_per_year: {Course.courses['year'].value_counts()}")
        #print(f"teacher course counts: {Course.courses['teacher'].value_counts()}")
        print(Course.courses)
        #for course in Course.course_list:
        #    print(f"Course {course.id}: Blocks {course.get_blocks}, Teacher: {course.teacher.id}, Year: {course.year}")

class Teacher:
    ids: list
    teachers: list = []

    def __init__(self, id: int):
        self.id = id
        self.courses : list[Course] = list()

    def add_course(self, c: Course):
        self.courses.append(c)

    @staticmethod
    def generate_teachers(n_teachers: int):
        Teacher.ids = list(range(0, n_teachers))
        for id in Teacher.ids:
            Teacher.teachers.append(Teacher(id))

class TimeSlot:
    slot_list: list = list()
    day: pd.DataFrame
    slot_array: np.ndarray
    ids: np.ndarray
    offs: np.ndarray

    def __init__(self, id: int, is_off: bool, course_index: int = None, cr_index: int = None):
        self.id = int(id)
        self.is_off = bool(is_off)
        self.course_index = course_index
        self.cr_index = cr_index

    @staticmethod
    def generate_day(n_slots: int, off_idxs: list[int]):
        TimeSlot.ids = np.arange(n_slots)
        TimeSlot.offs = np.full((n_slots,), fill_value=False, dtype=np.bool)
        TimeSlot.offs[off_idxs] = True
        TimeSlot.slot_array = np.column_stack([TimeSlot.ids, TimeSlot.offs])

        TimeSlot.slot_list = [TimeSlot(id, is_off) for id, is_off in zip(TimeSlot.ids, TimeSlot.offs)]
        TimeSlot.day = pd.DataFrame(TimeSlot.slot_array, columns=["id", "is_off"], index=None)
        #TimeSlot.display()

    def generate_week(n_days: int, n_slots_per_day: int, off_slot_lists_per_day):
        total_slots = n_days * n_slots_per_day
        TimeSlot.ids = np.arange(total_slots)
        TimeSlot.offs = np.zeros(total_slots, dtype=bool)

        for day in range(n_days):
            off_slots = off_slot_lists_per_day[day]
            # Convert from day-based slot to global index
            for s in off_slots:
                global_idx = day * n_slots_per_day + s
                TimeSlot.offs[global_idx] = True

        TimeSlot.slot_array = np.column_stack([TimeSlot.ids, TimeSlot.offs])
        TimeSlot.slot_list = [
            TimeSlot(id=idx, is_off=off)
            for idx, off in zip(TimeSlot.ids, TimeSlot.offs)
        ]
        TimeSlot.day = pd.DataFrame(TimeSlot.slot_array, columns=["id", "is_off"])
        TimeSlot.display()


    @staticmethod
    def get_available_blocks(block_size: int):
        blocks = []
        for start_idx in range(len(TimeSlot.slot_list) - block_size + 1):
            block = TimeSlot.slot_list[start_idx:start_idx + block_size]
            if all(not slot.is_off for slot in block):
                blocks.append(block)
        return blocks

    @staticmethod
    def display():
        print(TimeSlot.day)

def generate_targets(
        area: np.ndarray,
        seed: int = None,
        shape: tuple[int, int] = (50, 2),
        show = False
        ):
    deployment_bbox = area.reshape(-1, 2)
    X = np.empty(shape)
    n, d = shape

    for j in range(d):
        X[:, j] = np.round(np.random.uniform(deployment_bbox[j, 0], deployment_bbox[j, 1], size=n), 0)

    if show:
        plt.scatter(X[:, 0], X[:, 1], c='r', marker='*')
        plt.scatter(X[:, 0], X[:, 1], c='y', marker='+')
        plt.show()
    
    return X, area


def generate_bboxes(n, m, space_min=0, space_max=100):
    grid_size = int(np.sqrt(n))
    step = (space_max - space_min) / grid_size
    
    centers = [
        [space_min + (i + 0.5) * step, space_min + (j + 0.5) * step]
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    bounding_boxes = np.array([
        [x - m, x + m, y - m, y + m] for x, y in centers
    ])
    
    return np.array(centers), bounding_boxes


def distribute_rooms(room_ids, faculty_count):
    #random.shuffle(room_ids)
    faculty_rooms = [[] for _ in range(faculty_count)]
    
    chunk_size = len(room_ids) // faculty_count
    remainder = len(room_ids) % faculty_count
    
    start = 0
    for i in range(faculty_count):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        faculty_rooms[i] = room_ids[start:end]
        start = end
    
    return faculty_rooms

class Room:
    rooms: pd.DataFrame
    room_list: list = list()
    ids: np.ndarray
    capacities: np.ndarray
    room_array: np.ndarray
    room_locations: np.ndarray
    Faculty_ids: list = list()
    
    #day_idxs: np.ndarray = np.zeros_like(ids)
    # type_id : np.ndarray

    def __init__(
            self,
            id: int,
            capacity: int,
            location: np.ndarray,
            faculty_id: int
            # type_idx
    ):
        self.id = int(id)
        self.capacity = int(capacity)
        self.location = location
        self.faculty_id = faculty_id
        self.x = int(location[0])
        self.y = int(location[1])

    def generate_rooms(capacities: tuple[tuple[int, int]], faculty: int):
        m = 100
        centers, bboxes = generate_bboxes(faculty, 15, 0, 100)

        Room.ids = np.arange(0, sum([room_count for room_count, _ in capacities]))
        Room.capacities = np.hstack([[seat_count] * room_count for room_count, seat_count in capacities])
        np.random.shuffle(Room.capacities)
        
        faculty_rooms = distribute_rooms([int(id) for id in Room.ids], faculty_count=faculty)
        print(faculty_rooms)

        faculty_ids = []
        for id in Room.ids:
            for i, subset in enumerate(faculty_rooms):
                if id in subset:
                    faculty_ids.append(i)

        Room.Faculty_ids = faculty_ids

        rooms = []
        for i in range(faculty):
            area = bboxes[i].reshape(2, 2)
            room_locations, _ = generate_targets(area=bboxes[i], seed=None, shape=(len(faculty_rooms[i]), 2))
            rooms.append(room_locations)
        rooms = np.vstack(rooms)
        Room.room_locations = rooms

        Room.room_list = [Room(id, capacity, locations, faculty_id) for id, capacity, locations, faculty_id in zip(Room.ids, Room.capacities, Room.room_locations, faculty_ids)]
        Room.room_array = np.column_stack([Room.ids, Room.capacities])

        Room.rooms = pd.DataFrame({
            "id": Room.ids,
            "capacity": Room.capacities,
            "location": [list(loc) for loc in rooms],
            "faculty_id": faculty_ids
        })
        Room.display()

    def display():
        print(Room.rooms)
    
def display_results(R, T, x, solver, capacity=None, n_students=None):


    for (course, slot, room), var in x.items():
        if var.solution_value() == 1: 

            if capacity and n_students:
                cap = capacity(room)
                enroll = n_students(course)
                wasted = cap - enroll
                print(f"Assigned: Course {course}, Slot {slot}, Room {room} "
                      f"(Capacity={cap}, Students={enroll}, Wasted={wasted})")
            else:
                print(f"Assigned: Course {course}, Slot {slot}, Room {room}")

    
    rooms = R.ids    
    slots = T.ids     
    
    timetable = pd.DataFrame(0, index=slots, columns=rooms)

    for (course, slot, room), var in x.items():
        if var.solution_value() == 1:
            timetable.at[slot, room] = course+1
    
    import datetime
    time_indexes = [datetime.time(h+8, 30).strftime("%H:%M") for h in list(range(0, len(T.slot_list)))]
    print(time_indexes)
    timetable.index = time_indexes
    
    print("\nTime Slot / Room Assignment Table:")
    print(timetable)

def display_interval_results(courses, rooms, solver, 
                             start_vars, is_in_room_vars, block_size):
    """
    - courses: list of course objects
    - rooms: list of room objects
    - solver: the CpSolver
    - start_vars, is_in_room_vars: dictionaries linking (course, block, room)
      to the solver variables
    - block_size: dictionary or function giving the length of each block
    """
    for course in courses:
        for i, blk_size in enumerate(course.get_blocks):
            # Suppose we stored: start_vars[(course, i)]
            start_val = solver.Value(start_vars[(course.id, i)])
            assigned_room = None
            # figure out which room got is_in_room_vars=1
            for r in rooms:
                if solver.Value(is_in_room_vars[(course.id, i, r.id)]) == 1:
                    assigned_room = r.id
                    break
            
            print(f"Course {course.id}, Block {i}, "
                  f"starts at slot {start_val}, length={blk_size}, "
                  f"Room={assigned_room}")

from collections import defaultdict

def build_timetable(courses, rooms, horizon, solver, start_vars, is_in_room_vars, n_days):

    room_ids = [r.id for r in rooms]
    timetable = pd.DataFrame("", index=range(horizon), columns=room_ids)
    unique_rooms = defaultdict(list)

    for c in courses:
        for i, blk_size in enumerate(c.get_blocks):
            start_val = solver.Value(start_vars[(c.id, i)])

            for r in rooms:
                if solver.Value(is_in_room_vars[(c.id, i, r.id)]) == 1:
                    unique_rooms[r.id].append(blk_size)
 
                    for t in range(start_val, start_val + blk_size):
                        timetable.at[t, r.id] = f"C{c.id}({c.year}:{c.n_students}:{c.teacher.id})"

    print(f"Unique Rooms: {len(unique_rooms)}")
    times_used = []
    total_block_usage = []
    room_usage = {}
    for id, block_lengths in unique_rooms.items():
        times_used.append(len(block_lengths))
        total_block_usage.append(sum(block_lengths))
    
    print(f"number of rooms used: {len(times_used)}")
    print(f"sum times used: {(sum(times_used))}")
    print(f"sum total block usage: {(sum(total_block_usage))}")


    import datetime
    time_indexes = [datetime.time(h+8, 30).strftime("%H:%M") for h in list(range(9))]
    day_length = horizon // n_days  # e.g., if horizon=63 for 7 days, day_length=9
    day_tables = []


    for d in range(n_days):
        start_row = d * day_length
        end_row   = (d + 1) * day_length
        day_df = timetable.iloc[start_row:end_row, :].copy()
        day_df.index = time_indexes
        day_df.columns = [f"Rm{rooms[i].id}({rooms[i].capacity}[F{rooms[i].faculty_id}])" for i in range(len(rooms))]
        day_tables.append(day_df)
        
        

    # day_tables[d] is the timetable slice for day d
    return day_tables


def get_off_chunks(slot_list):
    """
    slot_list: a list of TimeSlot objects with attributes `.id` and `.is_off`.
    Returns a list of tuples (start, end) where each chunk of consecutive off-slots
    goes from 'start' to 'end' in discrete time.
    """
    off_chunks = []
    current_chunk = []
    
    # Sort slot_list by their .id just in case
    sorted_slots = sorted(slot_list, key=lambda s: s.id)
    
    for slot in sorted_slots:
        if slot.is_off:
            current_chunk.append(slot.id)
        else:
            # If we just ended a chunk, finalize it
            if len(current_chunk) > 0:
                first = current_chunk[0]
                last = current_chunk[-1]
                # chunk covers [first .. last+1)
                off_chunks.append((first, last+1))
                current_chunk = []
    
    # If there's a chunk at the very end
    if len(current_chunk) > 0:
        first = current_chunk[0]
        last = current_chunk[-1]
        off_chunks.append((first, last+1))
    
    return off_chunks


def find_subset_of_course(c: Course, subsets: list[list[Course]]):
    for i, subset in enumerate(subsets):
        if c in subset:
            return i
    raise Exception(f"Course {c.id} not found course_list")

def main_multi_day():
    num_days = 5
    slots_per_day = 9
    year = 4
    n_teachers = 9

    off_by_day = [[4] for _ in range(num_days)]
    off_by_day[-1] = off_by_day[-1] + [5]

    TimeSlot.generate_week(num_days, slots_per_day, off_by_day)
    Room.generate_rooms(((4, 48), (3, 36), (2, 72), (1, 90)), year)
    Teacher.generate_teachers(n_teachers)
    Course.generate_courses(28, (2,7), (15,90), Teacher.teachers)

    T = TimeSlot
    C = Course
    R = Room
    P = Teacher

    C_subsets = Course.years_subsets
    print(f"course_ids: {[c.id for subset in C_subsets for c in subset]}")
    print(f"C_subsets: {[c.year for subset in C_subsets for c in subset]}")

    model = cp_model.CpModel()
    horizon = num_days * slots_per_day

    all_intervals_per_room = {r.id: [] for r in Room.room_list}
    start_vars = {}
    is_in_rooms = {}


    all_intervals_for_subset = [[] for _ in C_subsets]
    all_intervals_for_teachers = [[] for _ in range(n_teachers)]
    seat_utilization_terms = [] # Obj func
    for c in Course.course_list:
        for i, block_size in enumerate(c.blocks):

            start_var = model.NewIntVar(0, horizon - block_size, f"start_c{c.id}_b{i}")
            end_var = model.NewIntVar(0, horizon, f"end_c{c.id}_b{i}")


            """
            Block Duration Constraint
            For eachg course block, this constraint ensures that the end time is exactly the
            start time plus the block'S fixed duration. In other word, every scheduled block of
            a course must have the correct length
            """
            model.Add(end_var == start_var + block_size)
            
            """
            Interval Variable Creation
            An interval variable is created for eachg course block. This varialbe represents the time
            span during which the course block takes place.
            """
            interval_var = model.NewIntervalVar(start_var, block_size, end_var,
                                                f"interval_c{c.id}_b{i}")
            
            c_subset_index = find_subset_of_course(c, C_subsets)
            all_intervals_for_subset[c_subset_index].append(interval_var)
            all_intervals_for_teachers[c.teacher.id].append(interval_var)

            in_room_bools = []
            for r in Room.room_list:
                in_r = model.NewBoolVar(f"inRoom_c{c.id}_b{i}_r{r.id}")

                """
                Capacity Check for Room Assignment Constraint
                For each room option, if a course block is assigned to that room (i.e. the boolean variable
                in_r is true), then the room's capacity must be at least as large as the number of students
                enrolled in the course
                """
                model.Add(r.capacity >= c.n_students).OnlyEnforceIf(in_r) # Encorce capacity(r) > n_students(c)
                in_room_bools.append(in_r)
                is_in_rooms[(c.id, i, r.id)] = in_r

                """
                Optional Interval per Room
                For each room, an optional interval is created that is "active" only when the course block is actually
                assigned to that room. These optional intervals will later be used to enforce that no two events in the
                same room overlap
                """
                opt_interval = model.NewOptionalIntervalVar(start_var, block_size, end_var,
                                                            in_r, f"optinterval_c{c.id}_b{i}_r{r.id}")
                all_intervals_per_room[r.id].append(opt_interval)

                """
                Objective function to minimize wasted space
                If a course block iis assigned to a room, this term adds the difference between the room's capacity and
                the number of students to the objective. Minimizing this term encourages assignments where the room's
                capacity closely fits the class size (i.e. reduces wasted space)
                """
                wasted_capacity = r.capacity - c.n_students
                seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_capacity)
                #wasted_ration = c.n_students / r.capacity
                #seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_ration)

            """
            Exactly-One Room Assignment Constraint
            For every course block, exactly one of the room assignment booleans must be true.
            This ensures that each block is scheduled in one-and only one- room
            """
            model.Add(sum(in_room_bools) == 1)
            start_vars[(c.id, i)] = start_var


    #model.Minimize(sum(seat_utilization_terms)) # Objective function
    off_chunks = get_off_chunks(TimeSlot.slot_list)

    day_vars = {}
    for c in C.course_list:
        for i, block_size in enumerate(c.get_blocks):

            # day_c_i in [0.. num_days - 1]
            day_c_i = model.NewIntVar(0, num_days - 1, f"day_c{c.id}_b{i}")
            day_vars[(c.id, i)] = day_c_i

            """
            Day Calculation from Start Time constraint
            The day on which a courseblock is scheduled is calculated by dividing its start time by the
            number of slots per day. This tells you on which dayt (0 to num_days - 1) the block begins
            """
            model.AddDivisionEquality(day_c_i, start_vars[(c.id, i)], slots_per_day)

            """
            Enforcing a Block Withing a Day constraint
            This constraint makes sure that a course block does not "spill over" into the next day by
            ensuring that the block finisihes before the day's end
            """
            model.Add(start_vars[(c.id, i)] + block_size <= (day_c_i + 1) * slots_per_day)
    
    for c in C.course_list:
        for d in range(num_days):
            blocks_in_day_lits = []
            for i, block_size in enumerate(c.get_blocks):
                day_c_i_d = model.NewBoolVar(f"day_c{c.id}_b{i}_d{d}")

                model.Add(day_vars[(c.id, i)] == d).OnlyEnforceIf(day_c_i_d)
                model.Add(day_vars[(c.id, i)] != d).OnlyEnforceIf(day_c_i_d.Not())

                blocks_in_day_lits.append(day_c_i_d)

            """
            At most One Block per Day per Course
            For every course, this constraint ensures that at most one block is scheduled on any given day
            (The booleans day_c_i_d are used to indicate whether a block is on day d)
            """    
            model.Add(sum(blocks_in_day_lits) <= 1)


    room_used = {r.id: model.NewBoolVar(f"room_used_{r.id}") for r in Room.room_list}
    for r in R.room_list:
        assignments_in_room = []
        for c in Course.course_list:
            for i, blk_size in enumerate(c.blocks):
                key = (c.id, i, r.id)
                if key in is_in_rooms:
                    assignments_in_room.append(is_in_rooms[key])
        """
        Room Usage Indicator
        For each room, a boolean variable room_used is set to 1 if any course block is assigned there.
        This is later used in the objective funcion to help minimize the number of rooms used.
        """
        model.AddMaxEquality(room_used[r.id], assignments_in_room)
    
    """
    Objective Term for Room Usage
    The sum of these terms (multiplied by a weight) is included in the objective to favor solutions that use fewer rooms
    """
    room_usage_terms = [room_used[r.id] for r in Room.room_list]

    central_x = {}
    central_y = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            """
            Central Coordiantes for Course Subsets
            For each course subset (for example, courses from the same year) and foır each day,
            these variables represents a "central" room location where you'd like the courses to cluster
            """
            central_x[(s, d)] = model.NewIntVar(0, 100, f"central_x_subset{s}_day{d}")
            central_y[(s, d)] = model.NewIntVar(0, 100, f"central_y_subset{s}_day{d}")
    
    # Come back hea
    abs_diff_x = {}
    abs_diff_y = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                abs_diff_x[(c.id, s, d)] = model.NewIntVar(0, 100, f"abs_diff_x_c{c.id}_sub{s}_day{d}")
                abs_diff_y[(c.id, s, d)] = model.NewIntVar(0, 100, f"abs_diff_y_c{c.id}_sub{s}_day{d}")
                # constraints
                for b, _ in enumerate(c.blocks):
                    for r in R.room_list:
                        """
                        Absolute Difference Between Room Location and Central Point
                        For every course in the subset, these constraints compute the absolute difference between
                        the room where the course block is scheduled and the central ocation for that subset on that day.
                        These differences are later summed to encourage courses to be geographically close to each other
                        """
                        model.Add(abs_diff_x[(c.id, s, d)] >= r.x * is_in_rooms[(c.id, b, r.id)] - central_x[(s, d)])
                        model.Add(abs_diff_x[(c.id, s, d)] >= central_x[(s, d)] - r.x * is_in_rooms[(c.id, b, r.id)])
                        model.Add(abs_diff_y[(c.id, s, d)] >= r.y * is_in_rooms[(c.id, b, r.id)] - central_y[(s, d)])
                        model.Add(abs_diff_y[(c.id, s, d)] >= central_y[(s, d)] - r.y * is_in_rooms[(c.id, b, r.id)])

    distance_terms = []
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                distance_terms.append(abs_diff_x[(c.id, s, d)] + abs_diff_y[(c.id, s, d)])
    #model.Minimize(sum(distance_terms))

    ### Minimize idle between subset c in day d
    earliest = {}
    latest = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            earliest[(s, d)] = model.NewIntVar(0, horizon, f"earliest_sub{s}_day{d}")
            latest[(s, d)] = model.NewIntVar(0, horizon, f"latest_sub{s}_day{d}")

    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            for c in subset:
                for b, block_size in enumerate(c.blocks):
                    """
                    Earliest and Latest Block Times per Subset per Day
                    For each subset of courses on each day, these constraints record the earliest starting time
                    and the latest finishing time among all blocks
                    """
                    model.Add(earliest[(s, d)] <= start_vars[(c.id, b)])
                    model.Add(latest[(s, d)] >= start_vars[(c.id, b)] + block_size)
    
    span = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            span[(s, d)] = model.NewIntVar(0, horizon, f"span_sub{s}_day{d}")

            """
            The "span" (the total time window during which the subset's classes occur) is computed.
            Minimizing this span would encourage classes for the same subset to be scheduled closer together,
            therby reducing idle gaps 
            """
            model.Add(span[(s, d)] == latest[(s, d)] - earliest[(s, d)])
    span_terms = [span[(s,d)] for s in range(len(C_subsets)) for d in range(num_days)]

    idle_time_weight = 2.0
    seat_utilization_weight = 8.0
    room_usage_weight = 2.0
    distance_weight = 5.0

    """
    Combined Objective
    Room Usage Term: Penalizes using many rooms by adding a cost for each room that is used
    Seat Utilization Term: Penalizes wasted capacity, encouraging courses to be assigned to rooms
    that fit them well. 
    """
    objective_expression = room_usage_weight * sum(room_usage_terms)
    objective_expression += seat_utilization_weight * sum(seat_utilization_terms)
    #objective_expression += distance_weight * sum(distance_terms)
    #objective_expression += idle_time_weight * sum(span_terms)
    model.Minimize(objective_expression) #Objective function
    #objective_expression = sum(room_usage_terms)
    #model.Minimize(sum(seat_utilization_terms)) #Objective function
    #model.Minimize(sum(room_used[r.id] for r in Room.room_list)*0.8) #Objective function
    

    for r in Room.room_list:
        for (start_off, end_off) in off_chunks:
            """
            Adding Off-Chunksas Intervals 
            For each room, certain time chunks are defined as "off" or unavailable. These intervals
            are added to the room's list of intervals so thast no course block can be scheduled during these times
            """
            off_int = model.NewIntervalVar(start_off, end_off - start_off, end_off,
                                           f"unavail_r{r.id}_{start_off}_{end_off}")
            all_intervals_per_room[r.id].append(off_int)

    for r_id, intervals in all_intervals_per_room.items():
        """
        Room No-Overlap
        For each room, all interavals ( both course blocks anmd the off periods) must not overlap.
        This ensures that no two events are scheduled in the same room at the same time. 
        """
        model.AddNoOverlap(intervals)

    for i in range(len(C_subsets)):
        """
        Subset No-Overlap
        For each subset of courases (e.g.by year), this constraint prevents overlapping
        intervals among the courses in that subset. This might be used to avoid conflicts
        for students who share the same cirriculum

        """
        model.AddNoOverlap(all_intervals_for_subset[i])

    for p in P.ids:
        """
        Teacher No-Overlap
        For each teacher, the model enforces that the teacher's assigned course intervals do not overlap,
        ensuring that a teacher isn't scheduled to be in two places at once.
        """
        model.AddNoOverlap(all_intervals_for_teachers[p])

    

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    """
        *   Course blocks have the proper length and are scheduled within day boundaries
        *   Each course block is assigned exactly one room that meets its capacity requirements
        *   Courses (especially those in the same subset) and teachers do not have overlapping time slots.
        *   Rooms are not double-booked and respect unavailable ("off") times.
        *   The schedule aims to minimize wasted room capacity and potentially minimizes distance and idle times.   
    """

    """ for c in C.course_list:
        for i, blk_size in enumerate(c.get_blocks):
            s_val = solver.Value(start_vars[(c.id, i)])
            d_val = solver.Value(day_vars[(c.id, i)])
            print(f"Course={c.id}, block={i}, start={s_val}, day={d_val}, subset={find_subset_of_course(c, C_subsets)}, teacher: {c.teacher.id}") """

    print(f"111 Number lessons: {C.courses['n_lessons'].sum()}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found (status={}):".format(status))

        timetable_df = build_timetable(
            courses=C.course_list,
            rooms=R.room_list,
            horizon=len(T.slot_list),
            solver=solver,
            start_vars=start_vars,
            is_in_room_vars=is_in_rooms,
            n_days=num_days
        )
        for df in timetable_df:
            print(df)
            print()
    else:
        print("No solution found (status={}).".format(status))

if __name__ == "__main__":
    seed = None
    np.random.seed(seed)
    random.seed(seed)   

    main_multi_day()
