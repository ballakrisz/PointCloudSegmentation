import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np 

OBJECT_PART_LABEL_RANGE = {
    'Airplane': list(range(4)),
    'Bag': list(range(4, 6)),
    'Cap': list(range(6, 8)),
    'Car': list(range(8, 12)),
    'Chair': list(range(12, 16)),
    'Earphone': list(range(16, 19)),
    'Guitar': list(range(19, 22)),
    'Knife': list(range(22, 24)),
    'Lamp': list(range(24, 28)),
    'Laptop': list(range(28, 30)),
    'Motorbike': list(range(30, 36)),
    'Mug': list(range(36, 38)),
    'Pistol': list(range(38, 41)),
    'Rocket': list(range(41, 44)),
    'Skateboard': list(range(44, 47)),
    'Table': list(range(47, 50))
}

PART_LABEL_COLORS = {
    0: 'purple',
    1: 'tan',
    2: 'blue',
    3: 'green',
    4: 'pink',
    5: 'red'
}

def visualize_points(point_cloud, part_label, object_label):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # create fig
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # split the data into groups based on the part label
    ss = np.array([x,y,z,part_label]).transpose(1,0)
    df = pd.DataFrame(ss, columns=['x', 'y', 'z', 'part_label'])
    grouped = df.groupby('part_label')
    grouped_data = {category: group[['x', 'y', 'z']].values for category, group in grouped}
    
    # plot the groups with their corresponding colors
    for category, group in grouped_data.items():
        object_label = object_label
        label_range = OBJECT_PART_LABEL_RANGE[object_label]
        idx = max(label_range) - category 
        ax.scatter(group[:,0], group[:,1], group[:,2], c=PART_LABEL_COLORS[idx], marker='o')

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(object_label)
    plt.show()