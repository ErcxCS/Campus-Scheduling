import pandas as pd
import numpy as np
from utils import midterm_timetable

class Department:
    departments = []
    department_names = []
    department_ids = []

    def __init__(self, department_name: str):
        self.courses = []
        self.curriculums = {i: [] for i in range(1, 5)}
        self.id = len(Department.departments)
        Department.department_ids.append(self.id)

        self.name = department_name
        words = self.name.split(" ")
        self.short = "".join(word[0] for word in words)
        Department.department_names.append(department_name)
        Department.departments.append(self)

    def add_course(self, course):
        if course not in self.courses:
            self.courses.append(course)
            self.curriculums.setdefault(course.year, []).append(course)

    def get_course(self, course_code: str):
        for course in self.courses:
            if course_code == course.course_code:
                return course
        return None

    @classmethod
    def get_dep(cls, dep_short: str):
        for dep in Department.departments:
            if dep_short == dep.short:
                return dep
        return None

    @classmethod
    def get_department(cls, department_name: str):
        if department_name in cls.department_names:
            idx = cls.department_names.index(department_name)
            return cls.departments[idx]
        return cls(department_name)


class Course:
    course_list = []
    course_codes = set()
    _by_code_and_inst = dict()
    pass_course_midterm = ["FİZ 176", "GIDA 324", "EEM 370"]

    def __init__(self, id: int, department: Department, course_name: str, year: int,
                 n_students: int, course_code: str, instructor_id: int,
                 requires_lab: bool, mandatory: str):
        self.id = int(id)
        self.n_students = int(n_students)
        self.departments = [department]
        self.course_code = course_code
        self.instructor_id = instructor_id
        self.requires_lab = requires_lab
        self.year = year
        self.course_name = course_name
        self.mandatory = True if mandatory == "Z" else False

        self.dep_short = ' '.join(dep.short for dep in self.departments)
        self.blocks = [2]

    def get_duration(self):
        return self.blocks[0]

    def get_dep_shorts(self):
        return ' '.join(dep.short for dep in self.departments)

    @staticmethod
    def read_courses(path: str, is_midterm: bool):
        df = pd.read_excel(path, index_col=None, header=0)
        
        # Use iterrows() instead of df.values
        for index, row in df.iterrows():
            department_name = row["DepartmentName"]
            course_code = row["CourseCode"]
            course_name = row["CourseName"]
            akts = row["AKTS"]
            credit = row["KREDI"]
            requires_lab = row["Lab"]
            year = row["SINIF"]
            mandatory = row["ZOR_SEC"]
            instructor_id = row["InstructorUserId"]
            n_students = row["OgrenciSayisi"]

            if is_midterm and course_code in Course.pass_course_midterm:
                continue

            dep = Department.get_department(department_name)
            inst_id = int(instructor_id)
            key = (course_code, inst_id)

            if key in Course._by_code_and_inst:
                existing = Course._by_code_and_inst[key]
                existing.departments.append(dep)
                existing.n_students += int(n_students)
                dep.add_course(existing)
            else:
                new_course = Course(
                    id=index, department=dep, course_name=course_name, year=int(year),
                    n_students=int(n_students), course_code=course_code,
                    instructor_id=inst_id, requires_lab=(requires_lab == 1),
                    mandatory=mandatory
                )
                dep.add_course(new_course)
                Course.course_codes.add(course_code)
                Course._by_code_and_inst[key] = new_course
                Course.course_list.append(new_course)


class TimeSlot:
    slot_list: list = []
    ids: np.ndarray
    offs: np.ndarray

    def __init__(self, id: int, is_off: bool):
        self.id = int(id)
        self.is_off = bool(is_off)

    @staticmethod
    def generate_week(n_days: int, n_slots_per_day: int, off_slot_lists_per_day):
        total_slots = n_days * n_slots_per_day
        TimeSlot.ids = np.arange(total_slots)
        TimeSlot.offs = np.zeros(total_slots, dtype=bool)

        for day in range(n_days):
            off_slots = off_slot_lists_per_day[day]
            for s in off_slots:
                global_idx = day * n_slots_per_day + s
                TimeSlot.offs[global_idx] = True

        TimeSlot.slot_list = [
            TimeSlot(id=idx, is_off=off)
            for idx, off in zip(TimeSlot.ids, TimeSlot.offs)
        ]


class Room:
    rooms: pd.DataFrame
    room_list: list = []
    ids: np.ndarray
    capacities: np.ndarray
    off_times_dict: dict = None
    special_ids: list = [int]
    special_rooms = ["Amfi 4", "Amfi 3", "BB01"]
    faculties = set()
    rooms_by_fac = dict()
    min_capacity = None

    def __init__(self, id: int, room_code: str, capacity: int, c_type: str, off_times: list = None):
        self.id = int(id)
        self.capacity = int(capacity)
        self.room_code = room_code
        self.is_lab = (c_type == "Lab")
        self.off_times = off_times
        self.faculty = room_code.split(sep="-")[0].strip()

        if room_code in Room.special_rooms:
            Room.special_ids.append(self.id)

    @staticmethod
    def read_classroom_data(path: str, num_days: int, slots_per_day: int, is_midterm: bool):
        df = pd.read_excel(path)
        df = df[df["Room"].notna()][["Room", "Capacity", "Type"]]

        room_ids = []
        room_codes = []
        room_caps = []
        labs = []
        regulars = []

        off_timetable = None
        if is_midterm:
            off_timetable = midterm_timetable(slots_per_day, num_days)
            Room.off_times_dict = off_timetable

        for i, row in enumerate(df.values):
            room_code, capacity, c_type = row
            room_ids.append(i)
            room_caps.append(int(capacity))
            room_codes.append(room_code)

            if is_midterm:
                room_obj = Room(i, room_code, int(capacity), c_type, off_timetable.get(room_code))
            else:
                room_obj = Room(i, room_code, int(capacity), c_type, None)

            if room_obj.is_lab:
                labs.append(room_obj)
            else:
                regulars.append(room_obj)
            
            Room.faculties.add(room_obj.faculty)
        
        Room.room_list = regulars + labs
        Room.ids = np.array(room_ids)
        Room.capacities = np.array(room_caps)
        Room.rooms = pd.DataFrame({
            "id": room_ids,
            "room_codes": room_codes,
            "capacities": room_caps
        })

        Room.rooms_by_fac = {fac: [] for fac in Room.faculties}
        Room.min_capacity = min(Room.capacities)
        for room in Room.room_list:
            Room.rooms_by_fac[room.faculty].append(room)

    @staticmethod
    def find_by_code(room_code: str):
        for room in Room.room_list:
            if room.room_code == room_code:
                return room
        raise Exception(f"room not found: {room_code} is not in room_list")