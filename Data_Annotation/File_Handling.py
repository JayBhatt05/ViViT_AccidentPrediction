class Node:
    def __init__(self, vehicle_id, vehicle_type):
        self.id = vehicle_id
        self.type = vehicle_type

def main():
    fptr = open("crash-1500-ann//000900_ann.txt", "r")
    new = open("annotations/000900.txt", "w")

    n = int(input("HOW MANY OBJECTS IN COLLISION--> "))
    arr = []

    for i in range(n):
        vehicle_id = int(input("ENTER ID--> "))
        vehicle_type = input("ENTER TYPE--> ")
        arr.append(Node(vehicle_id, vehicle_type))

    new.write("video_no\tFrame\tVehicle_ID\tVehicle_Type\tx1\ty1\twidth\theight\tAccident\n")

    # Skip the first 9 elements in the file
    for _ in range(9):
        fptr.readline()

    while True:
        temp_int = int(fptr.readline().strip())
        if temp_int == 0:
            break
        new.write(f"{temp_int:06d}\t")

        temp_int = int(fptr.readline().strip())
        new.write(f"{temp_int}\t")

        vehicle_id = int(fptr.readline().strip())
        new.write(f"{vehicle_id}\t")

        vehicle_type = fptr.readline().strip()
        new.write(f"{vehicle_type}\t")

        for _ in range(4):  # x1, y1, width, height
            temp_int = int(fptr.readline().strip())
            new.write(f"{temp_int}\t")

        accident_flag = int(fptr.readline().strip())

        ans = 0
        for vehicle in arr:
            if vehicle.id == vehicle_id and vehicle.type == vehicle_type:
                ans = 1
                break

        if ans == 1:
            new.write(f"1\n")
        else:
            new.write(f"{accident_flag}\n")

    fptr.close()
    new.close()

if __name__ == "__main__":
    main()
