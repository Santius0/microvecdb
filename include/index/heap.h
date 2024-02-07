#ifndef MICROVECDB_HEAP_H
#define MICROVECDB_HEAP_H

#include <vector>
#include <stdexcept>
#include <utility>

template<typename T>
class MinHeap {
private:
    std::vector<T> heap;

    void heapifyUp(T index) {
        while (index > 0 && heap[parent(index)] > heap[index]) {
            std::swap(heap[parent(index)], heap[index]);
            index = parent(index);
        }
    }

    void heapifyDown(T index) {
        T smallest = index;
        T leftChildIndex = leftChild(index);
        T rightChildIndex = rightChild(index);

        if (leftChildIndex < heap.size() && heap[leftChildIndex] < heap[smallest]) {
            smallest = leftChildIndex;
        }

        if (rightChildIndex < heap.size() && heap[rightChildIndex] < heap[smallest]) {
            smallest = rightChildIndex;
        }

        if (smallest != index) {
            std::swap(heap[index], heap[smallest]);
            heapifyDown(smallest);
        }
    }

    T parent(T index) { return (index - 1) / 2; }
    T leftChild(T index) { return 2 * index + 1; }
    T rightChild(T index) { return 2 * index + 2; }

public:
    MinHeap() = default;

    void push(T value) {
        heap.push_back(value);
        heapifyUp(heap.size() - 1);
    }

//    template<typename... Args>
//    void emplace(Args&&... args) {
//        heap.emplace_back(std::forward<Args>(args)...);
//        heapifyUp(heap.size() - 1);
//    }

    T top() {
        if (heap.empty()) {
            throw std::runtime_error("Heap is empty");
        }
        return heap[0];
    }

    T pop() {
        if (heap.empty()) {
            throw std::runtime_error("Heap is empty");
        }
        T min = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        heapifyDown(0);
        return min;
    }

    [[nodiscard]] bool isEmpty() const {
        return heap.empty();
    }

    size_t size(){
        return heap.size();
    }
};


#endif //MICROVECDB_HEAP_H
