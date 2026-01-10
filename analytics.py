import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from entities import Course, Room
from utils import read_fac_xlsxs, automated_df_rebuild

def get_concurrency_data(df, num_days):
    """
    Parses a dataframe and returns:
    1. room_overlap_map: {(day, slot, room): count}
    2. dept_concurrency_map: {(day, slot, dept): count}
    3. all_depts: set of all department names found
    """
    import ast
    
    SLOTS_PER_DAY = 9
    
    room_overlap_map = {} 
    dept_concurrency_map = {}
    all_depts = set()

    for _, row in df.iterrows():
        try:
            # 1. Parse Day/Slot (Handle variations)
            if 'Day' in row and pd.notna(row['Day']):
                day = int(row['Day']) - 1
            elif 'Date' in row and pd.notna(row['Date']): 
                day = int(row['Date']) - 1
            else: continue

            if 'Starting Slot' in row and pd.notna(row['Starting Slot']):
                start_slot = int(row['Starting Slot'])
            elif 'Slot' in row and pd.notna(row['Slot']):
                start_slot = int(row['Slot'])
            else: continue

            # 2. Parse Rooms
            rooms = []
            val = row.get('Assigned Rooms')
            if pd.notna(val):
                if str(val).startswith("["):
                    try: rooms = ast.literal_eval(str(val))
                    except: rooms = []
                else:
                    rooms = [r.strip() for r in str(val).split(',') if r.strip()]

            # 3. Parse Departments
            depts = []
            val_d = row.get('Departments')
            if pd.notna(val_d):
                depts = [d.strip() for d in str(val_d).split(',')]
                all_depts.update(depts)
            
            # 4. Fill Maps (assuming 2 slots per exam)
            for i in range(2):
                current_slot = start_slot + i
                if current_slot >= SLOTS_PER_DAY: break 
                
                for room in rooms:
                    key = (day, current_slot, room)
                    room_overlap_map[key] = room_overlap_map.get(key, 0) + 1
                
                for dep in depts:
                    key = (day, current_slot, dep)
                    dept_concurrency_map[key] = dept_concurrency_map.get(key, 0) + 1
                    
        except:
            continue
            
    return room_overlap_map, dept_concurrency_map, all_depts


def plot_concurrency_outputs(df_auto, df_man, num_days, save_dir):
    """
    Generates concurrency plots for both schedules using SHARED color limits.
    """
    import seaborn as sns
    
    SLOTS_PER_DAY = 9
    TOTAL_SLOTS = num_days * SLOTS_PER_DAY
    
    # 1. Get Data for Both
    auto_room, auto_dept, auto_depts_set = get_concurrency_data(df_auto, num_days)
    man_room, man_dept, man_depts_set = get_concurrency_data(df_man, num_days)
    
    # 2. Determine SHARED VMAX for Dept Heatmap
    # Find the highest number of simultaneous exams across both datasets
    max_concurrency_auto = max(auto_dept.values()) if auto_dept else 0
    max_concurrency_man = max(man_dept.values()) if man_dept else 0
    
    # Set shared upper limit (at least 1 to avoid crash)
    SHARED_VMAX = max(max_concurrency_auto, max_concurrency_man, 1)
    
    # 3. Define Plot Helper
    def plot_heatmap(dept_map, all_depts_set, title, vmax):
        sorted_depts = sorted(list(all_depts_set))
        n_depts = len(sorted_depts)
        if n_depts == 0: return

        dept_matrix = np.zeros((n_depts, TOTAL_SLOTS))
        for (d, s, dep), count in dept_map.items():
            global_slot = d * SLOTS_PER_DAY + s
            if dep in sorted_depts and global_slot < TOTAL_SLOTS:
                dept_idx = sorted_depts.index(dep)
                dept_matrix[dept_idx, global_slot] = count
        
        plt.figure(figsize=(18, max(4, len(sorted_depts) * 0.5)))
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        # KEY CHANGE: vmin=0, vmax=SHARED_VMAX
        sns.heatmap(dept_matrix, yticklabels=sorted_depts, cmap=cmap, 
                    linewidths=0.5, linecolor='gray', 
                    vmin=0, vmax=vmax,
                    cbar_kws={'label': 'Simultaneous Exams'})
        
        plt.title(f"{title}: Simultaneous Exams per Department (Max Scale: {vmax})")
        plt.xlabel("Time Slots")
        for d in range(1, num_days):
            plt.axvline(x=d * SLOTS_PER_DAY, color='black', linestyle='--', linewidth=1)
            
        save_path = os.path.join(save_dir, f"concurrency_depts_{title.lower()}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    def plot_hist(room_map, title):
        counts = list(room_map.values())
        if not counts: return
        max_val = max(counts)
        freq_bins = range(1, max_val + 2)
        
        plt.figure(figsize=(8, 5))
        plt.hist(counts, bins=freq_bins, align='left', rwidth=0.8, color='teal', edgecolor='black')
        plt.title(f"{title}: Room Sharing Frequency")
        plt.xlabel("Simultaneous Exams")
        plt.ylabel("Count")
        plt.xticks(freq_bins[:-1])
        plt.grid(axis='y', alpha=0.3)
        
        save_path = os.path.join(save_dir, f"concurrency_rooms_{title.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    # 4. Generate Plots
    print(f"Generating Dept Heatmaps with shared Max Limit: {SHARED_VMAX}")
    plot_heatmap(auto_dept, auto_depts_set, "Automated", SHARED_VMAX)
    plot_heatmap(man_dept, man_depts_set, "Manual", SHARED_VMAX)
    
    plot_hist(auto_room, "Automated")
    plot_hist(man_room, "Manual")


def plot_department_time_bias(df, title, save_dir):
    """
    Violin Plot to show 'Time-of-Day' bias per department.
    Answers: "Are some departments unfairly stuck with morning/evening exams?"
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Expand Data (Handle "CS, SE" -> separate rows)
    data = []
    
    for _, row in df.iterrows():
        try:
            # Check valid numeric Slot (0-8)
            if pd.isna(row['Starting Slot']): continue
            slot = int(row['Starting Slot'])
            
            # Parse Departments
            depts_str = str(row['Departments'])
            depts = [d.strip() for d in depts_str.split(',')]
            
            for dep in depts:
                data.append({'Department': dep, 'Slot': slot})
        except:
            continue

    if not data:
        print(f"No data available for {title} Time Bias plot.")
        return

    plot_df = pd.DataFrame(data)

    # 2. Filter: Only keep departments with enough exams to make a plot meaningful
    # (e.g., at least 3 exams)
    dept_counts = plot_df['Department'].value_counts()
    valid_depts = dept_counts[dept_counts >= 3].index
    plot_df = plot_df[plot_df['Department'].isin(valid_depts)]
    
    # Sort Departments alphabetically for easier reading
    plot_df = plot_df.sort_values('Department')

    if plot_df.empty:
        print(f"Not enough data per department for {title} Time Bias plot.")
        return

    # 3. Plotting
    # Height scales with number of departments
    plt.figure(figsize=(10, max(4, len(valid_depts) * 0.5)))
    
    # Violin Plot shows the density
    sns.violinplot(
        data=plot_df, 
        x="Slot", 
        y="Department", 
        inner="stick",  # Shows individual exam lines inside the violin
        density_norm="width", 
        linewidth=1,
        palette="coolwarm" # Cool colors (morning) -> Warm colors (evening) implied
    )
    
    plt.title(f"{title}: Departmental Time-of-Day Distribution\n(Left=Morning, Right=Evening)")
    plt.xlabel("Time Slot (0=Start of Day, 8=End of Day)")
    plt.ylabel("Department")
    plt.xlim(-1, 9) # Fixed range for 0-8 slots
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    save_path = os.path.join(save_dir, f"time_bias_{title.lower()}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_capacity_efficiency(df_auto, df_man, save_dir):
    """
    Scatter plot of Room Capacity vs Student Count.
    Helps visualize 'waste' (unused seats) and 'overcrowding'.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def get_data(df, source_name):
        # Filter valid rows
        # We need both Student Count and Capacity to be > 0
        valid = df[(df['Num Students'] > 0) & (df['Total Room Cap'] > 0)].copy()
        return valid['Total Room Cap'], valid['Num Students']

    # Extract Data
    cap_auto, stud_auto = get_data(df_auto, "Automated")
    cap_man, stud_man = get_data(df_man, "Manual")

    # Determine Axis Limits (Shared for fair comparison)
    max_cap = max(cap_auto.max() if not cap_auto.empty else 100, 
                  cap_man.max() if not cap_man.empty else 100)
    max_stud = max(stud_auto.max() if not stud_auto.empty else 100, 
                   stud_man.max() if not stud_man.empty else 100)
    
    limit = max(max_cap, max_stud) * 1.05

    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # Helper to draw the standard lines
    def draw_background(ax):
        # 100% Efficiency Line (y = x)
        ax.plot([0, limit], [0, limit], 'k--', alpha=0.5, label='100% Full')
        # 50% Efficiency Line (y = 0.5x)
        ax.plot([0, limit], [0, limit/2], 'k:', alpha=0.3, label='50% Full')
        
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Total Room Capacity")
        ax.set_ylabel("Number of Students")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')

    # --- Plot 1: Manual ---
    draw_background(axes[0])
    axes[0].scatter(cap_man, stud_man, color='gray', alpha=0.6, edgecolors='black', s=40)
    axes[0].set_title(f"Manual Schedule Efficiency\n(Points below diagonal = Wasted Seats)")

    # --- Plot 2: Automated ---
    draw_background(axes[1])
    axes[1].scatter(cap_auto, stud_auto, color='#d62728', alpha=0.6, edgecolors='black', s=40)
    axes[1].set_title(f"Automated Schedule Efficiency\n(Points below diagonal = Wasted Seats)")

    """ # Calculate and Display Average Utilization %
    if not cap_man.empty:
        util_man = (stud_man / cap_man).mean() * 100
        axes[0].text(limit*0.05, limit*0.9, f"Avg Fill: {util_man:.1f}%", 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    if not cap_auto.empty:
        util_auto = (stud_auto / cap_auto).mean() * 100
        axes[1].text(limit*0.05, limit*0.9, f"Avg Fill: {util_auto:.1f}%", 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.8)) """

    plt.tight_layout()
    save_path = os.path.join(save_dir, "capacity_efficiency_scatter.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def prepare_plot_data(df, num_days):
    """Parses DataFrame to get counts for plots."""
    exams_per_day = {d: 0 for d in range(num_days)}
    dep_year_per_day = {} # key: (day_idx, dep, year)

    for _, row in df.iterrows():
        try:
            if pd.isna(row['Day']): continue
            # Ensure Day is 0-indexed integer (Input is usually 1-based)
            day = int(row['Day']) - 1 
            if day < 0 or day >= num_days: continue
            
            # 1. Total Exams per day
            exams_per_day[day] = exams_per_day.get(day, 0) + 1
            
            # 2. Dep/Year stats
            deps_str = str(row['Departments'])
            year = int(row['Year'])
            
            # Handle "CS, SE" -> ["CS", "SE"]
            deps = [d.strip() for d in deps_str.split(',')]
            
            for dep in deps:
                key = (day, dep, year)
                dep_year_per_day[key] = dep_year_per_day.get(key, 0) + 1
                
        except (ValueError, TypeError):
            continue
            
    return exams_per_day, dep_year_per_day

def plot_comparison_exams_per_day(exams_man, exams_auto, num_days, save_path):
    x = list(range(1, num_days + 1))
    
    # Extract Y values for both (Day indices are 0-based in dict, 1-based in plot)
    y_man = [exams_man.get(day - 1, 0) for day in x]
    y_auto = [exams_auto.get(day - 1, 0) for day in x]
    
    plt.figure(figsize=(10, 5))
    
    # Plot Manual
    plt.plot(x, y_man, marker="o", linestyle='--', color='gray', label="Manual", alpha=0.7)
    
    # Plot Automated
    plt.plot(x, y_auto, marker="o", linestyle='-', color='#d62728', label="Automated", linewidth=2)
    
    plt.xlabel("Day")
    plt.ylabel("Number of Exams")
    plt.title(f"Daily Exam Load: Manual vs Automated")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def plot_dep_year_exam_counts(dep_year_per_day, num_days, title_suffix, save_path, shared_ymax=None):
    """
    Plots exam counts per day, separated by Department and Year.
    Supports a shared Y-axis limit for fair comparison.
    """
    import matplotlib.pyplot as plt

    keys = dep_year_per_day.keys()
    # Keys are (day, dep, year)
    dep_set = sorted({k[1] for k in keys}) 
    
    if not dep_set:
        print(f"No department data found for {title_suffix}")
        return

    # Dynamic height
    fig, axes = plt.subplots(len(dep_set), 1, figsize=(12, 4 * len(dep_set)), sharex=True)
    
    if len(dep_set) == 1:
        axes = [axes]
        
    years = [1, 2, 3, 4]
    year_colors = ["blue", "orange", "green", "purple"]

    for idx, dep_short in enumerate(dep_set):
        ax = axes[idx]
        
        # Determine local max for this specific department if no global max is provided
        local_max = 0
        
        for year in years:
            y_vals = [dep_year_per_day.get((day, dep_short, year), 0) for day in range(num_days)]
            if y_vals:
                local_max = max(local_max, max(y_vals))
            
            if sum(y_vals) > 0:
                ax.plot(range(1, num_days + 1), y_vals, marker="o", 
                        color=year_colors[year - 1], label=f"Year {year}")
        
        # APPLY SHARED Y-LIMIT HERE
        if shared_ymax is not None:
            # Add a small buffer (e.g. +1) so the highest point isn't cut off
            ax.set_ylim(0, shared_ymax + 1)
        else:
            # Fallback to local max with buffer
            ax.set_ylim(0, local_max + 1)
        
        ax.set_title(f"{dep_short} - {title_suffix}")
        ax.set_ylabel("Exam Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    
    plt.xlabel("Day")
    plt.xticks(range(1, num_days + 1))
    plt.suptitle(f"Exam Distribution by Department and Year ({title_suffix})", y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def mission_report(solver, start_vars, slots_per_day, in_room_vars, num_days):
    exams_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        exams_per_day[day_idx] = exams_per_day.get(day_idx, 0) + 1

    print("Exams scheduled per day:")
    total_exams = 0
    for day, count in sorted(exams_per_day.items()):
        print(f"  Day {day}: {count} exams")
        total_exams += count
    plot_exam_per_day(exams_per_day, num_days)
    print(f"Total exams: {total_exams}\n")

    dep_year_per_day = {}
    for e in Course.course_list:
        s_val = solver.Value(start_vars[e.id])
        day_idx = s_val // slots_per_day
        key = (day_idx, e.departments[0].short, e.year)
        dep_year_per_day[key] = dep_year_per_day.get(key, 0) + 1
    plot_dep_year_exam_counts(dep_year_per_day, num_days)

def find_unique_rooms(df: pd.DataFrame) -> list:
    all_rooms_list = df['Assigned Rooms'].str.split(',').explode().to_list()
    unique_rooms = {Room.find_by_code(code) for code in all_rooms_list}
    return list(unique_rooms)

def room_utilization(df: pd.DataFrame, is_manuel: bool = False):
    """
    Calculates room utilization metrics by day and overall.
    
    This version uses pandas' groupby for more efficient and readable
    processing, eliminating the need for nested explicit loops.
    """
    def calculate_total_capacity(df):
        rooms = find_unique_rooms(df)
        return sum(room.capacity if room.is_lab else room.capacity // 2 for room in rooms)
    
    def count_unique_rooms(df):
        return len(df['Assigned Rooms'].str.split(',').explode().unique())
        
    def find_flawed_entries_by_day(df):
        flawed_entries = df[df['Num Students'] > df['Total Room Cap']]
        return flawed_entries.groupby('Day').size()

    def calculate_excess_students_by_day(df):
        # Filter for entries where student count exceeds capacity
        excess_entries = df[df['Num Students'] > df['Total Room Cap']]
        # Calculate the excess students for each flawed entry
        excess_students = excess_entries['Num Students'] - excess_entries['Total Room Cap']
        # Group the excess students by day and sum them
        return excess_students.groupby(excess_entries['Day']).sum()

    if not is_manuel:
        slot_groups = df.groupby(['Day', 'Starting Slot'])
        slot_capacities = slot_groups.apply(calculate_total_capacity)
        slot_student_numbers = slot_groups['Num Students'].sum()
        slot_unique_rooms = slot_groups.apply(count_unique_rooms)
        daily_capacities = slot_capacities.groupby('Day').sum()
        daily_student_numbers = slot_student_numbers.groupby('Day').sum()
        unique_rooms_per_day = df.groupby('Day').apply(count_unique_rooms)
        daily_active_room_time_slots = slot_unique_rooms.groupby('Day').sum()
    else:
        daily_capacities = df.groupby('Day')['Total Room Cap'].sum()
        daily_student_numbers = df.groupby('Day')['Num Students'].sum()
        unique_rooms_per_day = df.groupby('Day')['Num Rooms Used'].sum()
        daily_active_room_time_slots = None
        
        flawed_entries_per_day = find_flawed_entries_by_day(df)
        print(f"Total number of infeasbile assignments: {sum(dict(flawed_entries_per_day).values())}")
        
        # Calculate and print the daily sum of excess students
        daily_excess_students = calculate_excess_students_by_day(df)
        print(f"Daily Sum of Excess Students: {dict(daily_excess_students)}")
        
        #daily_student_numbers = daily_student_numbers - daily_excess_students * 2

    daily_utilizations = daily_student_numbers / daily_capacities
    overall_utilization = daily_student_numbers.sum() / daily_capacities.sum()

    print(f"Daily Capacities: {list(daily_capacities)}")
    print(f"Daily Student Numbers: {list(daily_student_numbers)}")
    print(f"Overall Utilization: {overall_utilization}")
    print(f"Daily Utilizations: {list(daily_utilizations)}")
    
    if is_manuel:
        active_slots = sum(unique_rooms_per_day)
        print(f"Daily Active Room-Time Slots: {list(unique_rooms_per_day)}")
        print(f"Total Active Room-Time Slots: {sum(unique_rooms_per_day)}")
    else:
        active_slots = sum(daily_active_room_time_slots)
        print(f"Daily Active Room-Time Slots: {list(daily_active_room_time_slots)}")
        print(f"Total Active Room-Time Slots: {active_slots}")
        
    return active_slots, overall_utilization


def analyze_exam_distribution(df: pd.DataFrame, is_midterm: bool):
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')
    grouped = df_processed.groupby(['Departments', 'Year'])
    distribution_std_dev = {}

    for (department, year), group in grouped:
        daily_counts = group.groupby('Day').size()
        if is_midterm:
            relevant_days = range(1, 6) if year in [1, 3] else range(6, 11)
        else:
            num_days = int(df['Day'].max())
            relevant_days = range(1, num_days + 1)
        
        daily_counts = daily_counts.reindex(list(relevant_days), fill_value=0)
        distribution_std_dev[(department, year)] = np.std(daily_counts.values)

    return distribution_std_dev

def get_daily_exam_counts(df: pd.DataFrame, is_midterm: bool):
    df_processed = df.assign(Departments=df['Departments'].str.split(',')).explode('Departments')
    grouped = df_processed.groupby(['Departments', 'Year'])
    daily_counts_dict = {}
    num_days = int(df['Day'].max())
    full_index = list(range(1, num_days + 1))

    for (department, year), group in grouped:
        base_counts = group.groupby('Day').size()
        if is_midterm:
            relevant_days = range(1, 6) if year in [1, 3] else range(6, 11)
        else:
            relevant_days = full_index

        full_series = pd.Series(0, index=full_index, dtype=int)
        full_series.loc[list(relevant_days)] = base_counts.reindex(list(relevant_days), fill_value=0)
        daily_counts_dict[(department, year)] = full_series.to_list()

    return daily_counts_dict

def frequency_table2(experiment_no: int, exam: str, num_days):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import ast

    # ===============================
    # 1) LOAD BEAUTIFIED TIMETABLE
    # ===============================
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    timetable_xlsx = "beautified_exam_schedule.xlsx"
    timetable_path = os.path.join(exp_path, timetable_xlsx)
    timetable_df = pd.read_excel(timetable_path, index_col=None, header=0)

    # ===============================
    #  AUTOMATED TIMETABLE REBUILD
    # ===============================
    automated_xlsx = "faculty_schedule.xlsx"
    automated_path = os.path.join(exp_path, automated_xlsx)
    automated_df = pd.read_excel(automated_path, index_col=None, header=0)

    final_automated_df = automated_df_rebuild(automated_df, num_days)
    final_automated_df.to_excel("faculty_schedule_automated_finals.xlsx")

    # --- Sort automated timetable like manual ---
    automated_clean = final_automated_df.replace("", pd.NA)
    usage_automated = automated_clean.notna().sum(axis=0).sort_values(ascending=False)

    auto_sorted_cols = usage_automated.index
    auto_sorted_df = final_automated_df[auto_sorted_cols]

    auto_final_df = auto_sorted_df.T
    auto_final_df = auto_final_df.loc[
        auto_final_df.replace("", pd.NA).notna().sum(axis=1).sort_values(ascending=False).index
]

    auto_final_df.to_excel("faculty_schedule_automated_finals_sorted.xlsx")

    # Compute usage for beautified schedule
    cols = timetable_df.columns.drop("Day")
    usage_beautified = (
        timetable_df[cols]
        .notna()
        .sum()
        .sort_values(ascending=False)
    )

    # ========================================================
    # 2) REBUILD TIMETABLE FROM MANUAL FACULTY SCHEDULE
    # ========================================================
    path_in = "./data/B24_department_schedules_finals/faculty_schedule_manuel_finals.xlsx"
    path_out = "./faculty_schedule_manuel_finals_timetable.xlsx"

    df = pd.read_excel(path_in)

    # Parse Assigned Rooms string → Python list
    df["RoomsList"] = df["Assigned Rooms"].apply(ast.literal_eval)

    # Collect all unique rooms
    rooms = sorted({r for lst in df["RoomsList"] for r in lst})

    # Create timetable (64 rows, one per timeslot)
    n_rows = 8 * 8  # 8 days * 8 slots
    timetable = pd.DataFrame("", index=range(1, n_rows + 1), columns=rooms)

    # Track next free timeslot for each room
    room_next_row = {room: 1 for room in rooms}

    # Fill timetable
    for _, row in df.sort_values("Day").iterrows():
        course_id = row["Course ID"]
        for room in row["RoomsList"]:
            start = room_next_row[room]
            timetable.loc[start, room] = course_id
            timetable.loc[start + 1, room] = course_id
            room_next_row[room] += 2

    # ========================================================
    # 3) Collapse every 2 timeslots → 1 exam block
    # ========================================================
    merged = pd.DataFrame(columns=timetable.columns)

    for i in range(0, len(timetable), 2):
        block = {}
        for room in timetable.columns:
            top = timetable.iloc[i][room]
            bottom = timetable.iloc[i + 1][room]
            block[room] = top if top != "" else bottom
        merged.loc[i // 2] = block

    merged.index = range(1, len(merged) + 1)

    # ========================================================
    # 4) Sort columns (optional), transpose, sort rows by usage
    # ========================================================
    # Sort columns by usage before transpose
    sorted_cols = merged.notna().sum().sort_values(ascending=False).index
    merged_sorted = merged[sorted_cols]

    # Transpose -> rows are rooms
    final_df = merged_sorted.T

    # Treat empty strings as NaN
    clean = final_df.replace("", pd.NA)

    # Compute usage per room
    usage_generated = clean.notna().sum(axis=1)

    # Sort rooms by usage
    sorted_rows = usage_generated.sort_values(ascending=False).index
    final_df = final_df.loc[sorted_rows]

    # Save timetable
    final_df.to_excel(path_out, index_label="Room")

    # ========================================================
    # 5) TWO-WAY USAGE COMPARISON PLOT (Manual vs Automated)
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    max_y = max(
        usage_generated.max(),
        usage_automated.max()
    )

    # ------------------------------
    # Plot 1: Manual Schedule
    # ------------------------------
    axes[0].bar(
        usage_generated.sort_values(ascending=False).index,
        usage_generated.sort_values(ascending=False).values
    )
    axes[0].set_title("Manual Schedule — Room Usage")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_ylim(0, max_y)
    axes[0].set_xlabel("Rooms")
    axes[0].set_ylabel("Mission Count")

    # ------------------------------
    # Plot 2: Automated Schedule
    # ------------------------------
    axes[1].bar(usage_automated.index, usage_automated.values)
    axes[1].set_title("Automated Schedule — Room Usage")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_ylim(0, max_y)
    axes[1].set_xlabel("Rooms")
    axes[1].set_ylabel("Mission Count")

    plt.tight_layout()
    plt.show()


    import seaborn as sns
    import numpy as np


    # Convert strings → 1 (occupied) and empty → 0
    def to_binary(df):
        return df.replace("", np.nan).notna().astype(int)

    h_beautified = to_binary(timetable_df.drop(columns=["Day"]))
    h_manual     = to_binary(final_df)
    h_auto       = to_binary(auto_final_df)

    # --- Build mapping: room_code → "room_code (capacity)" ---
    room_label_map = {
        room.room_code: f"{room.room_code} ({room.capacity // 2 if not room.is_lab else room.capacity})"
        for room in Room.room_list
    }

    # --- Rename rows for manual + automated ---
    h_manual.index = [room_label_map.get(idx, idx) for idx in h_manual.index]
    h_auto.index   = [room_label_map.get(idx, idx) for idx in h_auto.index]

    # Plot 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    vmax = 1

    sns.heatmap(h_beautified, ax=axes[0], cmap="Blues", cbar=False, vmax=vmax)
    axes[0].set_title("Heatmap — My Method (Beautified)")
    axes[0].set_xlabel("Rooms")
    axes[0].set_ylabel("Timeslots / Blocks")

    sns.heatmap(h_manual, ax=axes[1], cmap="Greens", cbar=False, vmax=vmax)
    axes[1].set_title("Heatmap — Manual Schedule")
    axes[1].set_xlabel("Blocks")
    axes[1].set_ylabel("Rooms")

    sns.heatmap(h_auto, ax=axes[2], cmap="Reds", cbar=False, vmax=vmax)
    axes[2].set_title("Heatmap — Automated Schedule")
    axes[2].set_xlabel("Blocks")
    axes[2].set_ylabel("Rooms")

    # -----------------------------
    # KEEP SAME X-AXIS SCALE FOR MANUAL & AUTOMATED
    # -----------------------------

    # both must use the same number of columns
    max_x = max(h_manual.shape[1], h_auto.shape[1])

    # force same x-axis range and ticks
    for ax in [axes[1], axes[2]]:
        ax.set_xlim(0, max_x)
        ax.set_xticks(range(max_x))

    plt.tight_layout()
    plt.show()
    print("Timetable saved to:", path_out)

def frequency_table(experiment_no: int, exam: str, num_days: int):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ast
    from entities import Room

    # ===============================
    # 1. SETUP PATHS
    # ===============================
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    plots_path = os.path.join(exp_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    file_path = os.path.join(exp_path, "faculty_schedule2.xlsx")
    
    if not os.path.exists(file_path):
        print(f"[Error] Could not find {file_path}. Run analysis() first to generate merged data.")
        return

    df = pd.read_excel(file_path)

    # ===============================
    # 2. HELPER TO PARSE ROOMS
    # ===============================
    def parse_rooms(val):
        if pd.isna(val) or val == "":
            return []
        s = str(val).strip()
        if s.startswith("["):
            try:
                return ast.literal_eval(s)
            except:
                return []
        return [x.strip() for x in s.split(",") if x.strip()]

    # ===============================
    # 3. BUILD USAGE MATRICES
    # ===============================
    slots_per_day = 9 
    total_slots = num_days * slots_per_day
    
    room_obj_map = {r.room_code: r for r in Room.room_list}
    all_rooms_codes = sorted(room_obj_map.keys())
    room_to_idx = {code: i for i, code in enumerate(all_rooms_codes)}
    n_rooms = len(all_rooms_codes)

    # Binary Matrices (Rows=Rooms, Cols=TimeSlots)
    matrix_auto = np.zeros((n_rooms, total_slots), dtype=int)
    matrix_man = np.zeros((n_rooms, total_slots), dtype=int)

    for _, row in df.iterrows():
        # --- PROCESS AUTOMATED ---
        if pd.notna(row['Day']) and pd.notna(row['Starting Slot']):
            day = int(row['Day']) - 1
            slot = int(row['Starting Slot'])
            global_slot = day * slots_per_day + slot
            
            rooms_auto = parse_rooms(row['Assigned Rooms'])
            
            for r in rooms_auto:
                if r in room_to_idx:
                    idx = room_to_idx[r]
                    if global_slot < total_slots:
                        matrix_auto[idx, global_slot] = 1
                    if global_slot + 1 < total_slots:
                        matrix_auto[idx, global_slot + 1] = 1

        # --- PROCESS MANUAL ---
        if pd.notna(row.get('MAN_Day')) and pd.notna(row.get('MAN_Slot')):
            try:
                m_day = int(row['MAN_Day']) - 1
                m_slot = int(row['MAN_Slot'])
                global_m_slot = m_day * slots_per_day + m_slot
                
                rooms_man = parse_rooms(row.get('MAN_Assigned Rooms'))
                
                for r in rooms_man:
                    if r in room_to_idx:
                        idx = room_to_idx[r]
                        if global_m_slot < total_slots:
                            matrix_man[idx, global_m_slot] = 1
                        if global_m_slot + 1 < total_slots:
                            matrix_man[idx, global_m_slot + 1] = 1
            except:
                pass 

    # ===============================
    # 4. PLOT 1: ROOM USAGE BAR CHARTS (Side-by-Side)
    # ===============================
    
    # Calculate usage (blocks occupied)
    usage_auto_counts = {r: matrix_auto[room_to_idx[r]].sum() / 2 for r in all_rooms_codes}
    usage_man_counts = {r: matrix_man[room_to_idx[r]].sum() / 2 for r in all_rooms_codes}
    
    # --- Prepare Data ---
    sorted_rooms_man = sorted(all_rooms_codes, key=lambda r: usage_man_counts[r], reverse=True)
    y_man = [usage_man_counts[r] for r in sorted_rooms_man]
    x_man = np.arange(len(sorted_rooms_man))

    sorted_rooms_auto = sorted(all_rooms_codes, key=lambda r: usage_auto_counts[r], reverse=True)
    y_auto = [usage_auto_counts[r] for r in sorted_rooms_auto]
    x_auto = np.arange(len(sorted_rooms_auto))

    max_y = max(max(y_man) if y_man else 0, max(y_auto) if y_auto else 0)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Manual Bar Plot
    axes[0].bar(x_man, y_man, color='gray', alpha=0.7)
    axes[0].set_title('Manual Schedule — Room Usage')
    axes[0].set_ylabel('Exams Hosted')
    axes[0].set_ylim(0, max_y + 1)
    axes[0].set_xticks(x_man)
    axes[0].set_xticklabels(sorted_rooms_man, rotation=90, fontsize=8)

    # Automated Bar Plot
    axes[1].bar(x_auto, y_auto, color='#d62728', alpha=0.7)
    axes[1].set_title('Automated Schedule — Room Usage')
    axes[1].set_ylabel('Exams Hosted')
    axes[1].set_ylim(0, max_y + 1)
    axes[1].set_xticks(x_auto)
    axes[1].set_xticklabels(sorted_rooms_auto, rotation=90, fontsize=8)

    bar_path = os.path.join(plots_path, "room_usage.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved Bar Chart: {bar_path}")

    # ===============================
    # 5. PLOT 2: HEATMAPS (Sorted Density Style)
    # ===============================
    
    # Helper to convert Matrix -> Sorted DataFrame with Labels
    def prepare_heatmap_df(matrix, usage_dict):
        # 1. Convert to DataFrame
        df_temp = pd.DataFrame(matrix, index=all_rooms_codes)
        
        # 2. Filter out completely empty rooms to save space
        active_rooms = [r for r in all_rooms_codes if usage_dict[r] > 0]
        if not active_rooms:
            return pd.DataFrame() # Empty result
            
        df_temp = df_temp.loc[active_rooms]
        
        # 3. Sort by Usage (Busiest on Top)
        df_temp['sum'] = df_temp.sum(axis=1)
        df_temp = df_temp.sort_values('sum', ascending=False)
        df_temp = df_temp.drop(columns=['sum'])
        
        # 4. Rename Index to include Capacity
        new_index = []
        for code in df_temp.index:
            r_obj = room_obj_map.get(code)
            cap = r_obj.capacity if r_obj.is_lab else r_obj.capacity // 2
            new_index.append(f"{code} ({cap})")
        df_temp.index = new_index
        
        return df_temp

    # Prepare DataFrames
    h_manual = prepare_heatmap_df(matrix_man, usage_man_counts)
    h_auto = prepare_heatmap_df(matrix_auto, usage_auto_counts)

    if h_manual.empty and h_auto.empty:
        print("No data available for heatmaps.")
        return

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    vmax = 1

    # Manual Heatmap (Green)
    if not h_manual.empty:
        sns.heatmap(h_manual, ax=axes[0], cmap="Greens", cbar=False, vmax=vmax)
        axes[0].set_title("Heatmap — Manual Schedule")
        axes[0].set_xlabel("Time Slots")
        axes[0].set_ylabel("Rooms")
        
        # Add Day Lines
        for d in range(1, num_days):
            axes[0].axvline(x=d * slots_per_day, color='black', linestyle='--', linewidth=0.5)

    # Automated Heatmap (Red)
    if not h_auto.empty:
        sns.heatmap(h_auto, ax=axes[1], cmap="Reds", cbar=False, vmax=vmax)
        axes[1].set_title("Heatmap — Automated Schedule")
        axes[1].set_xlabel("Time Slots")
        axes[1].set_ylabel("Rooms")
        
        # Add Day Lines
        for d in range(1, num_days):
            axes[1].axvline(x=d * slots_per_day, color='black', linestyle='--', linewidth=0.5)

    # Match X-Axis Scale for fair comparison
    max_cols = total_slots # The matrices are fixed size
    for ax in axes:
        ax.set_xlim(0, max_cols)

    plt.tight_layout()
    heatmap_path = os.path.join(plots_path, "heatmaps_comparison.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved Heatmaps: {heatmap_path}")

def merge_dfs(fac_df_manuel: pd.DataFrame, fac_df_out: pd.DataFrame, path) -> pd.DataFrame:
    import ast # Ensure ast is available for safe parsing if needed

    # --- Helper to force "A,B" format ---
    def normalize_rooms(val):
        # 1. If it's already a Python list, join it
        if isinstance(val, list):
            return ",".join(val)
        
        # 2. If it's a string looking like "['A', 'B']", parse and join
        s = str(val).strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return ",".join(parsed)
            except:
                pass
        
        # 3. Otherwise assume it's already "A,B" or empty
        return s

    # Apply normalization to the Manual dataframe column
    if 'Assigned Rooms' in fac_df_manuel.columns:
        fac_df_manuel['Assigned Rooms'] = fac_df_manuel['Assigned Rooms'].apply(normalize_rooms)

    # --- Existing Merge Logic ---
    sort_keys = ['Departments', 'Course ID']
    fac_df_manuel_sorted = fac_df_manuel.sort_values(sort_keys).reset_index(drop=True)
    fac_df_out_sorted = fac_df_out.sort_values(sort_keys).reset_index(drop=True)
    
    cols_to_append = ['Day', 'Slot', 'Assigned Rooms', 'Num Rooms Used', 'Total Room Cap', 'Num Students']
    man_cols = fac_df_manuel_sorted[cols_to_append].rename(columns={c: f"MAN_{c}" for c in cols_to_append})
    merged = pd.concat([fac_df_out_sorted, man_cols], axis=1)
    merged.to_excel(path, index=False)
    return merged


def analysis(experiment_no: int, is_midterm: bool, num_days: int):
    # Setup Paths
    exam = "midterms" if is_midterm else "finals"
    exp_path = f"./runs/exp{experiment_no}_{exam[:-1]}"
    plots_path = os.path.join(exp_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    # 1. Load Raw Data & Create Merge
    fac_df_manuel, fac_df_out = read_fac_xlsxs(experiment_no, exam)
    
    faculty_xlsx = "faculty_schedule2.xlsx"
    faculty_path = os.path.join(exp_path, faculty_xlsx)
    merged = merge_dfs(fac_df_manuel, fac_df_out, faculty_path)

    # ==========================================
    # 2. FREQUENCY & HEATMAP ANALYSIS
    # ==========================================
    frequency_table(experiment_no, exam, num_days)

    # ==========================================
    # 3. PREPARE DATA FROM MERGED DF
    # ==========================================
    
    # View A: Automated
    df_auto_view = merged.copy() 

    # View B: Manual
    man_cols_map = {
        'MAN_Day': 'Day',
        'MAN_Slot': 'Starting Slot',
        'MAN_Assigned Rooms': 'Assigned Rooms',
        
        # Stats Columns needed for Efficiency Plot
        'MAN_Total Room Cap': 'Total Room Cap', 
        'MAN_Num Students': 'Num Students',
        
        # Shared Columns
        'Departments': 'Departments', 
        'Year': 'Year'
    }
    # Create the manual view dataframe
    available_cols = [c for c in man_cols_map.keys() if c in merged.columns]
    df_man_view = merged[available_cols].rename(columns=man_cols_map)
    
    # Filter out empty manual rows
    df_man_view = df_man_view.dropna(subset=['Day', 'Starting Slot'])
    # ==========================================
    # 4. RUN DISTRIBUTION PLOTS
    # ==========================================
    print("\n--- Generating Distribution Plots ---")
    
    epd_auto, dy_auto = prepare_plot_data(df_auto_view, num_days)
    epd_man, dy_man = prepare_plot_data(df_man_view, num_days)

    plot_comparison_exams_per_day(
        epd_man, 
        epd_auto, 
        num_days, 
        os.path.join(plots_path, "exams_per_day_comparison.png")
    )

# --- CALCULATE SHARED Y-MAX ---
    # Find the highest number of exams scheduled for any (Day, Dept, Year) tuple
    # across both datasets.
    max_val_auto = max(dy_auto.values()) if dy_auto else 0
    max_val_man = max(dy_man.values()) if dy_man else 0
    
    # The limit should be the highest peak found in either schedule
    SHARED_YMAX = max(max_val_auto, max_val_man)
    
    # Pass this limit to both functions
    plot_dep_year_exam_counts(dy_auto, num_days, "Automated", 
                              os.path.join(plots_path, "dep_dist_automated.png"),
                              shared_ymax=SHARED_YMAX)
    
    plot_dep_year_exam_counts(dy_man, num_days, "Manual", 
                              os.path.join(plots_path, "dep_dist_manual.png"),
                              shared_ymax=SHARED_YMAX)
    # ==========================================
    # 5. CONCURRENCY & OVERLAP ANALYSIS
    # ==========================================
    print("\n--- Generating Concurrency Plots ---")
    
    plot_concurrency_outputs(df_auto_view, df_man_view, num_days, plots_path)

    # ==========================================
    # 6. TEXT STATISTICAL ANALYSIS
    # ==========================================
    print("\n--- Generating Efficiency Plots ---")
    plot_capacity_efficiency(df_auto_view, df_man_view, plots_path)

    print("--- Room Utilization Analysis (Automated) ---")
    act_ts_au, util_au = room_utilization(fac_df_out)
    print("\n--- Room Utilization Analysis (Manual) ---")
    act_ts_ma, util_ma = room_utilization(fac_df_manuel, is_manuel=True)
    print("-" * 30)

    room_time_improvement = 100 - (act_ts_au / act_ts_ma) * 100
    room_utilization_improvement = 100 - (util_ma / util_au) * 100

    print(f"\nRoom-Time Improvement: {room_time_improvement:.2f}%")
    print(f"Room-Utilization Improvement: {room_utilization_improvement:.2f}%")
    print("-" * 30)

    print("\n--- Exam Distribution Analysis ---")
    dist_manual = analyze_exam_distribution(fac_df_manuel, is_midterm)
    dist_auto = analyze_exam_distribution(fac_df_out, is_midterm)

    avg_std_manual = np.mean(list(dist_manual.values()))
    avg_std_auto = np.mean(list(dist_auto.values()))
    
    distribution_improvement = 100 - (avg_std_auto / avg_std_manual) * 100

    print(f"Average Standard Deviation of Daily Exams (Manual): {avg_std_manual:.2f}")
    print(f"Average Standard Deviation of Daily Exams (Automated): {avg_std_auto:.2f}")
    print(f"Improvement in Exam Distribution Uniformity: {distribution_improvement:.2f}%")
