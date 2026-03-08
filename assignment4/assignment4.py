# Name - Win Yu Maung
# ID - 6612054
# Sec - 541


def merge(arr, temp, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inversions = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            inversions += mid - i + 1
            j += 1
        k += 1

    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1

    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1

    index = left
    while index <= right:
        arr[index] = temp[index]
        index += 1

    return inversions


def merge_sort(arr, temp, left, right):
    if left >= right:
        return 0

    mid = (left + right) // 2
    left_count = merge_sort(arr, temp, left, mid)
    right_count = merge_sort(arr, temp, mid + 1, right)
    split_count = merge(arr, temp, left, mid, right)

    return left_count + right_count + split_count


def count_inversions(arr):
    n = len(arr)
    if n == 0:
        return 0
    temp = [0] * n
    return merge_sort(arr, temp, 0, n - 1)


def read_next_number():
    line = input()
    while line == "":
        line = input()
    return int(line)


def main():
    t = read_next_number()

    for _ in range(t):
        n = read_next_number()
        arr = []
        for _ in range(n):
            arr.append(int(input()))

        print(count_inversions(arr))

if __name__ == "__main__":
    main()
