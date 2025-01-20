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
        """
        off_slot_lists_per_day: e.g. a list of length n_days,
        where each entry is a list of slot indices that are off for that day.
        """
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
        """
        Get all possible consecutive blocks of `block_size` in the schedule.
        """
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
    """
    R: The Room object (containing R.ids, etc.)
    T: The TimeSlot object (containing T.ids, etc.)
    x: Dict of variables keyed by (course, slot, room).
    solver: The cp_model.CpSolver() or similar.
    capacity: Optional function or dict to get a room's capacity. e.g. capacity(room_id).
    n_students: Optional function or dict to get a course's enrollment. e.g. n_students(course_id).
    """

    # 1) Print out each assigned (course, slot, room)
    for (course, slot, room), var in x.items():
        if var.solution_value() == 1:  # CP-SAT: solver.Value(...)
            # If you have capacity and n_students, you can print extra info:
            if capacity and n_students:
                cap = capacity(room)
                enroll = n_students(course)
                wasted = cap - enroll
                print(f"Assigned: Course {course}, Slot {slot}, Room {room} "
                      f"(Capacity={cap}, Students={enroll}, Wasted={wasted})")
            else:
                print(f"Assigned: Course {course}, Slot {slot}, Room {room}")

    # 2) Create a timetable DataFrame with rows=slots and columns=rooms
    rooms = R.ids      # e.g. np.array([0,1,2]) or [1,2,3]
    slots = T.ids      # e.g. np.array([0,1,2,3,4,5,...])
    
    timetable = pd.DataFrame(0, index=slots, columns=rooms)

    # 3) Fill the DataFrame: Put the course ID (or course+1 if 0-based) in each used (slot, room)
    for (course, slot, room), var in x.items():
        if var.solution_value() == 1:
            # If your course IDs start at 0, 
            # you might prefer to write "course + 1" in the table to make it 1-based:
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
    # Create an empty DataFrame with rows=0..(horizon-1), columns=room IDs
    room_ids = [r.id for r in rooms]
    timetable = pd.DataFrame("", index=range(horizon), columns=room_ids)
    unique_rooms = defaultdict(list)
    # Fill the big timetable
    for c in courses:
        for i, blk_size in enumerate(c.get_blocks):
            start_val = solver.Value(start_vars[(c.id, i)])
            # Which room?
            for r in rooms:
                if solver.Value(is_in_room_vars[(c.id, i, r.id)]) == 1:
                    unique_rooms[r.id].append(blk_size)
                    # Fill [start_val..start_val+blk_size-1]
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



    # Now split timetable into n_days parts
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


def main2():
    Course.generate_courses(4, (3, 8), (15, 32))
    TimeSlot.generate_day(9, 4)
    Room.generate_rooms(((1, 28), (2, 32)))

    # Create a CP-SAT model
    model = cp_model.CpModel()

    # Sets
    T = TimeSlot  # T slots in the day, e.g. 9 "on" slots
    R = Room      # R rooms
    C = Course    # C courses


    # 1) Prepare data structures
    all_intervals_per_room = {r.id: [] for r in R.room_list}
    start_vars = {}
    is_in_rooms = {}

    # 2) Create intervals for each course-block
    for c in C.course_list:
        blocks = c.get_blocks
        for i, block_size in enumerate(blocks):
            start_var = model.NewIntVar(0, len(T.slot_list) - block_size,
                                        f"start_c{c.id}_b{i}")
            end_var   = model.NewIntVar(0, len(T.slot_list), 
                                        f"end_c{c.id}_b{i}")
            
            # Link end = start + block_size
            model.Add(end_var == start_var + block_size)
            
            # "Base" interval (not optional). We'll pair it with optional intervals:
            interval_var = model.NewIntervalVar(
                start_var, block_size, end_var,
                f"interval_c{c.id}_b{i}"
            )
            
            in_room_bools = []
            
            for room_obj in R.room_list:
                in_r = model.NewBoolVar(f"inRoom_c{c.id}_b{i}_r{room_obj.id}")
                in_room_bools.append(in_r)
                is_in_rooms[(c.id, i, room_obj.id)] = in_r
                
                # Optional interval
                opt_interval_var = model.NewOptionalIntervalVar(
                    start_var, block_size, end_var,
                    in_r,  # active if in_r == 1
                    f"optinterval_c{c.id}_b{i}_r{room_obj.id}"
                )
                
                # Add to that room's intervals
                all_intervals_per_room[room_obj.id].append(opt_interval_var)
            
            # Force exactly 1 chosen room
            model.Add(sum(in_room_bools) == 1)
            
            # Store the start var if you'll need it for building a timetable
            start_vars[(c.id, i)] = start_var

    # 3) Build and add "off" intervals (for each "off chunk") **once** per room
    off_chunks = get_off_chunks(T.slot_list)
    for room_obj in R.room_list:
        r_id = room_obj.id
        for (start_off, end_off) in off_chunks:
            off_interval = model.NewIntervalVar(
                start_off,
                end_off - start_off,
                end_off,
                f"unavail_r{r_id}_{start_off}_{end_off}"
            )
            all_intervals_per_room[r_id].append(off_interval)

    # 4) No Overlap in each room
    for r_id, intervals in all_intervals_per_room.items():
        model.AddNoOverlap(intervals)



    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found (status={}):".format(status))
        # Build a timetable DataFrame
        timetable_df = build_timetable(
            courses=C.course_list,
            rooms=R.room_list,
            horizon=len(T.slot_list),  # e.g. 9
            solver=solver,
            start_vars=start_vars,
            is_in_room_vars=is_in_rooms,
            block_size_func=None  # We just do course.blocks[i]
        )
        print(timetable_df)
    else:
        print("No solution found (status={}).".format(status))

    
def main():
    Course.generate_courses(5, (3, 5), (15, 32))
    TimeSlot.generate_day(9, 4)
    Room.generate_rooms(((1, 28), (2, 32)))

    # Create a CP-SAT model
    #model = pywraplp.Solver.CreateSolver("SCIP")
    model = pywraplp.Solver.CreateSolver("SCIP")

    # Sets
    C = Course
    T = TimeSlot
    R = Room

    # Parameters
    n_slots = lambda course_id: int(C.courses[C.courses["id"] == course_id]["n_lessons"].values) # number of time slots t required by lesson of course c
    #n_slots = lambda course_id: max(C.course_list[course_id].blocks)
    n_students = lambda course_id: int(C.courses[C.courses["id"] == course_id]["n_students"].values) # number of students enrolled in course c
    capacity = lambda room_id: int(R.rooms[R.rooms["id"] == room_id]["capacity"].values) # capacity of room r

   
    # Decision variable
    x = {}
    for course in C.course_list:
        for slot in T.slot_list:
            for room in R.room_list:
                #x[(course.id, slot.id, room.id)] = model.NewBoolVar(f"x_c{course.id}_t{slot.id}_r{room.id}")
                x[(course.id, slot.id, room.id)] = model.BoolVar(f"x_c{course.id}_t{slot.id}_r{room.id}")

    y = {}
    
    for slot in T.get_available_blocks(3):
        print(len(slot))

    # Constraints

    for course in C.course_list:
        for room in R.room_list:
            for slot in T.slot_list:
                if slot.is_off:
                    model.Add(x[(course.id, slot.id, room.id)] == 0)
                else:
                    model.Add(x[(course.id, slot.id, room.id)] <= 1)

    for course in C.course_list:
        constraint_slot = []
        block_size = max(course.blocks)
        #block_size = n_slots(course.id)

        for t, slot in enumerate(T.slot_list):
            """ if t + block_size > len(T.slot_list) - 1:
                print(t + block_size)
                print(len(T.slot_list) - 1)
                continue

            block = T.slot_list[t : t + block_size]
            if any(slot.is_off for slot in block):
                continue
             """
            if not slot.is_off:
                constraint_room = []
                for room in R.room_list:
                    constraint_room.append(x[(course.id, slot.id, room.id)])
                    constraint_slot.append(x[(course.id, slot.id, room.id)])
                model.Add(sum(constraint_room) <=1)
        model.Add(sum(constraint_slot) == block_size)

    for course in C.course_list:
        constraint_cons = []
        block_size = max(course.blocks)

        for room in R.room_list:
            constaint_per_room = []
            for slot in T.slot_list:
                if not slot.is_off:
                    constaint_per_room.append(x[(course.id, slot.id, room.id)])
                    constraint_cons.append(x[(course.id, slot.id, room.id)])
            model.Add(sum(constaint_per_room) <= block_size)
        model.Add(sum(constraint_cons) == block_size)

    for room in R.room_list:
        for slot in T.slot_list:
            if not slot.is_off:
                constraint_overlap = []
                for course in C.course_list:
                    constraint_overlap.append(x[(course.id, slot.id, room.id)])
                model.Add(sum(constraint_overlap) <= 1)


    #solver = cp_model.CpSolver()
    #status = solver.Solve(model)
    status = model.Solve()

    if status == model.OPTIMAL:
        print("Optimal solution found")
        display_results(R, T, x, model)

    elif status == model.FEASIBLE:
        print("Feasible solution found!")
        display_results(R, T, x, model)
    else:
        print("No solution found.")

def main3():
    Course.generate_courses(6, (3, 5), (15, 32))
    TimeSlot.generate_day(9, 4)
    Room.generate_rooms(((1, 28), (2, 32)))

    # Create a CP-SAT model
    #model = pywraplp.Solver.CreateSolver("SCIP")
    model = pywraplp.Solver.CreateSolver("SCIP")

    # Sets
    C = Course
    T = TimeSlot
    R = Room

    for c in C.course_list:
        for block in c.get_blocks:
            print(block)

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


    # Suppose off_by_day is a list of length 7,
    # each element is a list of "off slot indices" for that day
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

    # Flatten horizon = num_days * slots_per_day
    horizon = num_days * slots_per_day

    # same approach:
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
            model.Add(end_var == start_var + block_size)
            
            interval_var = model.NewIntervalVar(start_var, block_size, end_var,
                                                f"interval_c{c.id}_b{i}")
            
            c_subset_index = find_subset_of_course(c, C_subsets)
            all_intervals_for_subset[c_subset_index].append(interval_var)
            all_intervals_for_teachers[c.teacher.id].append(interval_var)

            in_room_bools = []
            for r in Room.room_list:
                in_r = model.NewBoolVar(f"inRoom_c{c.id}_b{i}_r{r.id}")
                model.Add(r.capacity >= c.n_students).OnlyEnforceIf(in_r) # Encorce capacity(r) > n_students(c)
                in_room_bools.append(in_r)
                is_in_rooms[(c.id, i, r.id)] = in_r

                opt_interval = model.NewOptionalIntervalVar(start_var, block_size, end_var,
                                                            in_r, f"optinterval_c{c.id}_b{i}_r{r.id}")
                all_intervals_per_room[r.id].append(opt_interval)

                # Objective function to minimize wasted space
                wasted_capacity = r.capacity - c.n_students
                seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_capacity)
                #wasted_ration = c.n_students / r.capacity
                #seat_utilization_terms.append(is_in_rooms[(c.id, i, r.id)] * wasted_ration)

            
            model.Add(sum(in_room_bools) == 1)
            start_vars[(c.id, i)] = start_var


    #model.Minimize(sum(seat_utilization_terms)) # Objective function
    # Build off_chunks with day-based logic => just returns a list of (start_off, end_off) in [0..horizon)
    off_chunks = get_off_chunks(TimeSlot.slot_list)

    day_vars = {}
    for c in C.course_list:
        for i, block_size in enumerate(c.get_blocks):

            # day_c_i in [0.. num_days - 1]
            day_c_i = model.NewIntVar(0, num_days - 1, f"day_c{c.id}_b{i}")
            day_vars[(c.id, i)] = day_c_i

            model.AddDivisionEquality(day_c_i, start_vars[(c.id, i)], slots_per_day)
            model.Add(start_vars[(c.id, i)] + block_size <= (day_c_i + 1) * slots_per_day)
    
    for c in C.course_list:
        for d in range(num_days):
            blocks_in_day_lits = []
            for i, block_size in enumerate(c.get_blocks):
                day_c_i_d = model.NewBoolVar(f"day_c{c.id}_b{i}_d{d}")

                model.Add(day_vars[(c.id, i)] == d).OnlyEnforceIf(day_c_i_d)
                model.Add(day_vars[(c.id, i)] != d).OnlyEnforceIf(day_c_i_d.Not())

                blocks_in_day_lits.append(day_c_i_d)
            model.Add(sum(blocks_in_day_lits) <= 1)


    room_used = {r.id: model.NewBoolVar(f"room_used_{r.id}") for r in Room.room_list}
    for r in R.room_list:
        assignments_in_room = []
        for c in Course.course_list:
            for i, blk_size in enumerate(c.blocks):
                key = (c.id, i, r.id)
                if key in is_in_rooms:
                    assignments_in_room.append(is_in_rooms[key])
        model.AddMaxEquality(room_used[r.id], assignments_in_room)
    room_usage_terms = [room_used[r.id] for r in Room.room_list]

    central_x = {}
    central_y = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            central_x[(s, d)] = model.NewIntVar(0, 100, f"central_x_subset{s}_day{d}")
            central_y[(s, d)] = model.NewIntVar(0, 100, f"central_y_subset{s}_day{d}")
    
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
                    model.Add(earliest[(s, d)] <= start_vars[(c.id, b)])
                    model.Add(latest[(s, d)] >= start_vars[(c.id, b)] + block_size)
    
    span = {}
    for s, subset in enumerate(C_subsets):
        for d in range(num_days):
            span[(s, d)] = model.NewIntVar(0, horizon, f"span_sub{s}_day{d}")
            model.Add(span[(s, d)] == latest[(s, d)] - earliest[(s, d)])
    span_terms = [span[(s,d)] for s in range(len(C_subsets)) for d in range(num_days)]

    idle_time_weight = 2.0
    seat_utilization_weight = 8.0
    room_usage_weight = 2.0
    distance_weight = 5.0

    objective_expression = room_usage_weight * sum(room_usage_terms)
    objective_expression += seat_utilization_weight * sum(seat_utilization_terms)
    objective_expression += distance_weight * sum(distance_terms)
    #objective_expression += idle_time_weight * sum(span_terms)
    model.Minimize(objective_expression) #Objective function
    #objective_expression = sum(room_usage_terms)
    #model.Minimize(sum(seat_utilization_terms)) #Objective function
    #model.Minimize(sum(room_used[r.id] for r in Room.room_list)*0.8) #Objective function
    

    for r in Room.room_list:
        for (start_off, end_off) in off_chunks:
            off_int = model.NewIntervalVar(start_off, end_off - start_off, end_off,
                                           f"unavail_r{r.id}_{start_off}_{end_off}")
            all_intervals_per_room[r.id].append(off_int)

    for r_id, intervals in all_intervals_per_room.items():
        model.AddNoOverlap(intervals)

    for i in range(len(C_subsets)):
        model.AddNoOverlap(all_intervals_for_subset[i])

    for p in P.ids:
        model.AddNoOverlap(all_intervals_for_teachers[p])

    

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    """ for c in C.course_list:
        for i, blk_size in enumerate(c.get_blocks):
            s_val = solver.Value(start_vars[(c.id, i)])
            d_val = solver.Value(day_vars[(c.id, i)])
            print(f"Course={c.id}, block={i}, start={s_val}, day={d_val}, subset={find_subset_of_course(c, C_subsets)}, teacher: {c.teacher.id}") """

    print(f"111 Number lessons: {C.courses['n_lessons'].sum()}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found (status={}):".format(status))
        # Build a timetable DataFrame
        timetable_df = build_timetable(
            courses=C.course_list,
            rooms=R.room_list,
            horizon=len(T.slot_list),  # e.g. 9
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

    #main()
    #main2()
    #main3()
    main_multi_day()
