# Open demand XML file
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

font = {"size": 22}
matplotlib.rc("font", **font)

fig, ax = plt.subplots()
fig.set_size_inches(16, 10, forward=True)
ax.set_xlabel("Minute")
ax.set_ylabel("Vehicles (vpm)")
ax.set_title("Peak Hour Demand")

demand_files = ["University Blvd & S State St.xml", "400S & 200W.xml"]
for demand_file in demand_files:
    departures = defaultdict(int)
    with open(demand_file, "r") as f:
        demand = ET.parse(f)
        root = demand.getroot()
        for child in root:
            if child.tag == "vehicle":
                depart = int(child.attrib["depart"])
                departures[depart] += 1

    first_depart = list(sorted(departures))[0]
    departs_per_min = defaultdict(int)

    minute = 0
    for depart in range(first_depart, 3600 + first_depart):
        count = departures[depart]
        second = depart - first_depart
        if second % 60 == 0:
            minute += 1
        print(minute, second)
        departs_per_min[int(minute)] += count

    ax.plot(
        list(departs_per_min.keys()),
        list(departs_per_min.values()),
        label=demand_file[:-4],
    )

plt.legend()
fig.tight_layout()
plt.show()
