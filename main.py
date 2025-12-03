import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

sys.setrecursionlimit(10000)
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class AVLTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class RBTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.color = "RED"

class BST:
    def __init__(self):
        self.root = None

    def search(self, value):
        x = self.root
        while x is not None:
            if value == x.value:
                return True
            elif value < x.value:
                x = x.left
            else:
                x = x.right
        return False

    def insert(self, value):
        new_node = TreeNode(value)
        if self.root is None:
            self.root = new_node
            return

        x = self.root
        while True:
            if value < x.value:
                if x.left is None:
                    x.left = new_node
                    return
                else:
                    x = x.left
            elif value > x.value:
                if x.right is None:
                    x.right = new_node
                    return
                else:
                    x = x.right
            else:
                return

    def delete(self, value):
        parent = None
        x = self.root

        # Ищем узел для удаления и его родителя
        while x is not None and x.value != value:
            parent = x
            if value < x.value:
                x = x.left
            else:
                x = x.right

        if x is None:
            return  # Value not found

        # Случай 1: У удаляемого узла нет потомков
        if x.left is None and x.right is None:
            if x == self.root:
                self.root = None
            elif parent.left == x:
                parent.left = None
            else:
                parent.right = None

        # Случай 2: У удаляемого узла один потомок
        elif x.left is None or x.right is None:
            if x.left is not None:
                child = x.left
            else:
                child = x.right

            if x == self.root:
                self.root = child
            elif parent.left == x:
                parent.left = child
            else:
                parent.right = child

        # Случай 3: У удаляемого узла два потомка
        else:
            # Находим минимальный узел в правом поддереве
            suc_parent = x
            suc = x.right
            while suc.left is not None:
                suc_parent = suc
                suc = suc.left

            # Заменяем значение удаляемого узла на значение преемника
            x.value = suc.value

            # Удаляем преемника
            if suc_parent.left == suc:
                suc_parent.left = suc.right
            else:
                suc_parent.right = suc.right

    def find_max(self):
        if self.root is None:
            return None
        x = self.root
        while x.right is not None:
            x = x.right
        return x.value

    def find_min(self):
        if self.root is None:
            return None
        x = self.root
        while x.left is not None:
            x = x.left
        return x.value

    def pre_order_traversal(self):
        self._pre_order_recursive(self.root)

    def _pre_order_recursive(self, x):
        if x is None:
            return
        print(x.value, end=" ")
        self._pre_order_recursive(x.left)
        self._pre_order_recursive(x.right)

    def in_order_traversal(self):
        self._in_order_recursive(self.root)

    def _in_order_recursive(self, x):
        if x is None:
            return
        self._in_order_recursive(x.left)
        print(x.value, end=" ")
        self._in_order_recursive(x.right)

    def post_order_traversal(self):
        self._post_order_recursive(self.root)

    def _post_order_recursive(self, x):
        if x is None:
            return
        self._post_order_recursive(x.left)
        self._post_order_recursive(x.right)
        print(x.value, end=" ")

    def level_order_traversal(self):
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            x = queue.pop(0)
            print(x.value, end=" ")
            if x.left:
                queue.append(x.left)
            if x.right:
                queue.append(x.right)

    def get_height(self):
        if self.root is None:
            return 0

        height = 0
        queue = [(self.root, 1)]  # (node, level)

        while queue:
            node, level = queue.pop(0)
            if level > height:
                height = level

            if node.left:
                queue.append((node.left, level + 1))
            if node.right:
                queue.append((node.right, level + 1))

        return height


class AVLTree:
    def __init__(self):
        self.root = None

    def search(self, value):
        x = self.root
        while x is not None:
            if value == x.value:
                return True
            elif value < x.value:
                x = x.left
            else:
                x = x.right
        return False

    def _get_height(self, x):
        if x is None:
            return 0
        return x.height

    def _get_balance(self, x):
        if x is None:
            return 0
        return self._get_height(x.left) - self._get_height(x.right)

    def _update_height(self, x):
        if x is not None:
            x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))

    def _left_rotate(self, n):
        k = n.right
        T = k.left

        k.left = n
        n.right = T

        self._update_height(n)
        self._update_height(k)

        return k

    def _right_rotate(self, n):
        k = n.left
        T = k.right

        k.right = n
        n.left = T

        self._update_height(n)
        self._update_height(k)

        return k

    def _balance_node(self, x):
        if x is None:
            return x

        self._update_height(x)
        balance = self._get_balance(x)

        # LL (малый правый поворот)
        if balance > 1 and self._get_balance(x.left) >= 0:
            return self._right_rotate(x)

        # RR (малый левый поворот)
        if balance < -1 and self._get_balance(x.right) <= 0:
            return self._left_rotate(x)

        # LR (большой лево-правый поворот)
        if balance > 1 and self._get_balance(x.left) < 0:
            x.left = self._left_rotate(x.left)
            return self._right_rotate(x)

        # RL (большой право-левый поворот)
        if balance < -1 and self._get_balance(x.right) > 0:
            x.right = self._right_rotate(x.right)
            return self._left_rotate(x)

        return x

    def _find_min_node(self, x):
        cur = x
        while cur.left is not None:
            cur = cur.left
        return cur

    def insert(self, value):
        self.root = self._insert_recursive(self.root, value)

    def _insert_recursive(self, x, value):
        if x is None:
            return AVLTreeNode(value)

        if value < x.value:
            x.left = self._insert_recursive(x.left, value)
        elif value > x.value:
            x.right = self._insert_recursive(x.right, value)
        else:
            return x  # Дубликат не вставляем

        return self._balance_node(x)

    def delete(self, value):
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, x, value):
        if x is None:
            return x

        if value < x.value:
            x.left = self._delete_recursive(x.left, value)
        elif value > x.value:
            x.right = self._delete_recursive(x.right, value)
        else:
            # Нашли узел для удаления
            if x.left is None:
                return x.right
            elif x.right is None:
                return x.left
            else:
                # Узел с двумя потомками
                temp = self._find_min_node(x.right)
                x.value = temp.value
                x.right = self._delete_recursive(x.right, temp.value)

        return self._balance_node(x)

    def find_max(self):
        if self.root is None:
            return None
        x = self.root
        while x.right is not None:
            x = x.right
        return x.value

    def find_min(self):
        if self.root is None:
            return None
        x = self.root
        while x.left is not None:
            x = x.left
        return x.value

    def pre_order_traversal(self):
        self._pre_order_recursive(self.root)

    def _pre_order_recursive(self, x):
        if x is None:
            return
        print(x.value, end=" ")
        self._pre_order_recursive(x.left)
        self._pre_order_recursive(x.right)

    def in_order_traversal(self):
        self._in_order_recursive(self.root)

    def _in_order_recursive(self, x):
        if x is None:
            return
        self._in_order_recursive(x.left)
        print(x.value, end=" ")
        self._in_order_recursive(x.right)

    def post_order_traversal(self):
        self._post_order_recursive(self.root)

    def _post_order_recursive(self, x):
        if x is None:
            return
        self._post_order_recursive(x.left)
        self._post_order_recursive(x.right)
        print(x.value, end=" ")

    def level_order_traversal(self):
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            x = queue.pop(0)
            print(x.value, end=" ")
            if x.left:
                queue.append(x.left)
            if x.right:
                queue.append(x.right)

    def get_height(self):
        """Возвращает высоту дерева"""
        return self._get_height(self.root) if self.root else 0

class RBTree:
    def __init__(self):
        self.NIL = RBTreeNode(None)
        self.NIL.color = "BLACK"
        self.NIL.left = self.NIL
        self.NIL.right = self.NIL
        self.root = self.NIL

    def _create_node(self, value):
        node = RBTreeNode(value)
        node.left = self.NIL
        node.right = self.NIL
        node.parent = None
        node.color = "RED"
        return node

    def _is_red(self, node):
        return node != self.NIL and node.color == "RED"

    def _is_black(self, node):
        return node == self.NIL or node.color == "BLACK"

    def search(self, value):
        x = self.root
        while x != self.NIL:
            if value == x.value:
                return True
            elif value < x.value:
                x = x.left
            else:
                x = x.right
        return False

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left

        if y.left != self.NIL:
            y.left.parent = x

        y.parent = x.parent

        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right

        if y.right != self.NIL:
            y.right.parent = x

        y.parent = x.parent

        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y

        y.right = x
        x.parent = y

    def transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v

        v.parent = u.parent

    def _find_min_node(self, x):
        while x.left != self.NIL:
            x = x.left
        return x

    def find_max(self):
        if self.root == self.NIL:
            return None
        x = self.root
        while x.right != self.NIL:
            x = x.right
        return x.value

    def find_min(self):
        if self.root == self.NIL:
            return None
        x = self.root
        while x.left != self.NIL:
            x = x.left
        return x.value

    def insert(self, value):
        new_node = self._create_node(value)

        parent = None
        cur = self.root

        while cur != self.NIL:
            parent = cur
            if value < cur.value:
                cur = cur.left
            elif value > cur.value:
                cur = cur.right
            else:
                return

        new_node.parent = parent

        if parent is None:
            self.root = new_node
        elif value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        self._insert_fixup(new_node)

    def _insert_fixup(self, node):
        while node != self.root and self._is_red(node.parent):
            parent = node.parent
            grandparent = parent.parent

            if parent == grandparent.left:
                uncle = grandparent.right

                # Случай 1: Красный дядя
                if self._is_red(uncle):
                    parent.color = "BLACK"
                    uncle.color = "BLACK"
                    grandparent.color = "RED"
                    node = grandparent
                    continue

                # Случай 2: Чёрный дядя, node - правый ребёнок
                if node == parent.right:
                    self._left_rotate(parent)
                    node = parent
                    parent = node.parent

                # Случай 3: Чёрный дядя, node - левый ребёнок
                parent.color = "BLACK"
                grandparent.color = "RED"
                self._right_rotate(grandparent)
                break

            else:
                uncle = grandparent.left

                if self._is_red(uncle):
                    parent.color = "BLACK"
                    uncle.color = "BLACK"
                    grandparent.color = "RED"
                    node = grandparent
                    continue

                if node == parent.left:
                    self._right_rotate(parent)
                    node = parent
                    parent = node.parent

                parent.color = "BLACK"
                grandparent.color = "RED"
                self._left_rotate(grandparent)
                break

        self.root.color = "BLACK"

    def delete(self, value):
        # Ищем удаляемый узел z
        z = self.root
        while z != self.NIL and z.value != value:
            if value < z.value:
                z = z.left
            else:
                z = z.right

        if z == self.NIL:
            return

        y = z
        y_original_color = y.color

        if z.left == self.NIL:
            x = z.right
            self.transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self._find_min_node(z.right)
            y_original_color = y.color
            x = y.right

            if y.parent != z:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

            x.parent = y

        # Если удалённый узел был ЧЁРНЫМ → фиксируем
        if y_original_color == "BLACK":
            self._delete_fixup(x)

    def _delete_fixup(self, x):
        while x != self.root and x.color == "BLACK":
            if x == x.parent.left:
                sibling = x.parent.right

                # Случай 1: брат красный
                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    x.parent.color = "RED"
                    self._left_rotate(x.parent)
                    sibling = x.parent.right

                # Случай 2: оба ребенка брата черные
                if sibling.left.color == "BLACK" and sibling.right.color == "BLACK":
                    sibling.color = "RED"
                    x = x.parent
                else:
                    # Случай 3: левый ребенок брата красный, правый черный
                    if sibling.right.color == "BLACK":
                        sibling.left.color = "BLACK"
                        sibling.color = "RED"
                        self._right_rotate(sibling)
                        sibling = x.parent.right

                    # Случай 4: правый ребенок брата красный
                    sibling.color = x.parent.color
                    x.parent.color = "BLACK"
                    sibling.right.color = "BLACK"
                    self._left_rotate(x.parent)
                    x = self.root
            else:

                sibling = x.parent.left

                if sibling.color == "RED":
                    sibling.color = "BLACK"
                    x.parent.color = "RED"
                    self._right_rotate(x.parent)
                    sibling = x.parent.left

                if sibling.right.color == "BLACK" and sibling.left.color == "BLACK":
                    sibling.color = "RED"
                    x = x.parent
                else:
                    if sibling.left.color == "BLACK":
                        sibling.right.color = "BLACK"
                        sibling.color = "RED"
                        self._left_rotate(sibling)
                        sibling = x.parent.left

                    sibling.color = x.parent.color
                    x.parent.color = "BLACK"
                    sibling.left.color = "BLACK"
                    self._right_rotate(x.parent)
                    x = self.root

        x.color = "BLACK"

    def pre_order_traversal(self):
        if self.root == self.NIL:
            return
        self._pre_order_recursive(self.root)

    def _pre_order_recursive(self, x):
        if x == self.NIL:
            return
        print(x.value, end=" ")
        self._pre_order_recursive(x.left)
        self._pre_order_recursive(x.right)

    def in_order_traversal(self):
        if self.root == self.NIL:
            return
        self._in_order_recursive(self.root)

    def _in_order_recursive(self, x):
        if x == self.NIL:
            return
        self._in_order_recursive(x.left)
        print(x.value, end=" ")
        self._in_order_recursive(x.right)

    def post_order_traversal(self):
        if self.root == self.NIL:
            return
        self._post_order_recursive(self.root)

    def _post_order_recursive(self, x):
        if x == self.NIL:
            return
        self._post_order_recursive(x.left)
        self._post_order_recursive(x.right)
        print(x.value, end=" ")

    def level_order_traversal(self):
        if self.root == self.NIL:
            return
        queue = [self.root]
        while queue:
            x = queue.pop(0)
            print(x.value, end=" ")
            if x.left != self.NIL:
                queue.append(x.left)
            if x.right != self.NIL:
                queue.append(x.right)

    def get_height(self):
        if self.root == self.NIL:
            return 0

        height = 0
        queue = [(self.root, 1)]  # (node, level)

        while queue:
            node, level = queue.pop(0)
            if level > height:
                height = level

            if node.left != self.NIL:
                queue.append((node.left, level + 1))
            if node.right != self.NIL:
                queue.append((node.right, level + 1))

        return height


def run_experiment(tree_class, values_generator, max_n=1000, step=50, num_trials=5):
    sizes = list(range(step, max_n + 1, step))
    heights = []

    for n in sizes:
        print(f"  Обрабатываем {n} узлов...", end="\r")
        trial_heights = []

        for trial in range(num_trials):
            try:
                values = values_generator(n)
                tree = tree_class()

                for val in values:
                    tree.insert(val)

                trial_heights.append(tree.get_height())
            except Exception as e:
                print(f"\nОшибка при n={n}, trial={trial}: {e}")
                continue

        if trial_heights:
            heights.append(np.mean(trial_heights))
        else:
            heights.append(0)
            print(f"\nНе удалось получить данные для n={n}")

    print()
    return sizes, heights


def generate_random_values(n):
    return random.sample(range(n * 10), n)


def generate_sorted_values(n):
    return list(range(n))


def plot_bst_random(sizes, heights):
    """1 BST(случайные ключи)"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, heights, 'b-o', linewidth=2, markersize=4, label='Экспериментальные данные')
    c = estimate_bst_constant(sizes, heights)

    log_fit = [c * math.log2(n + 1) for n in sizes]
    plt.plot(sizes, log_fit, 'g--', linewidth=1.5,
             label=f'Логарифмическая аппроксимация: {c:.2f}·log₂(n)')

    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Высота дерева h(n)')
    plt.title('Экспериментальная зависимость высоты BST от количества ключей\n' +
              'Случайные равномерно распределенные ключи')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    textstr = (
        'Наблюдаемая асимптотика: O(log n)\n'
        f'Экспериментальная константа: c ≈ {c:.2f}\n'
        'Теория: средняя высота случайного BST ≈ 4.311·ln(n) ≈ 6.22·log₂(n)'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)

    plt.show()

def estimate_bst_constant(sizes, heights):
    log_n = [math.log2(n) for n in sizes]
    x = np.array(log_n)
    y = np.array(heights)
    coeff = np.polyfit(x, y, 1)
    c = coeff[0]
    return c

def plot_avl_random(sizes, heights):
    """2 AVL(случайные ключи)"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, heights, 'r-', linewidth=2, label='AVL (эксперимент)')
    upper_bound = [1.44 * math.log2(n + 1) for n in sizes]
    lower_bound = [math.log2(n + 1) for n in sizes]

    plt.plot(sizes, upper_bound, 'r--', alpha=0.7,
             label='Верхняя граница: 1.44·log₂(n)')
    plt.plot(sizes, lower_bound, 'r:', alpha=0.7,
             label='Нижняя граница: log₂(n+1)')

    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Высота дерева h(n)')
    plt.title('AVL: Высота от количества узлов (случайные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rb_random(sizes, heights):
    """3 RB (случайные ключи)"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, heights, 'g-', linewidth=2, label='RB (эксперимент)')
    upper_bound = [2.0 * math.log2(n + 1) for n in sizes]
    lower_bound = [math.log2(n + 1) for n in sizes]

    plt.plot(sizes, upper_bound, 'g--', alpha=0.7,
             label='Верхняя граница: 2·log₂(n+1)')
    plt.plot(sizes, lower_bound, 'g:', alpha=0.7,
             label='Нижняя граница: log₂(n+1)')

    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Высота дерева h(n)')
    plt.title('Красно-черное дерево: Высота от количества узлов (случайные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_avl_sorted(sizes, heights):
    """4 AVL (отсортированные ключи)"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, heights, 'r-', linewidth=2, label='AVL (эксперимент)')
    upper_bound = [1.44 * math.log2(n + 1) for n in sizes]
    lower_bound = [math.log2(n + 1) for n in sizes]

    plt.plot(sizes, upper_bound, 'r--', alpha=0.7,
             label='Верхняя граница: 1.44·log₂(n)')
    plt.plot(sizes, lower_bound, 'r:', alpha=0.7,
             label='Нижняя граница: log₂(n+1)')

    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Высота дерева h(n)')
    plt.title('AVL: Высота от количества узлов (отсортированные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rb_sorted(sizes, heights):
    """5 RB (отсортированные ключи)"""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, heights, 'g-', linewidth=2, label='RB (эксперимент)')
    upper_bound = [2.0 * math.log2(n + 1) for n in sizes]
    lower_bound = [math.log2(n + 1) for n in sizes]

    plt.plot(sizes, upper_bound, 'g--', alpha=0.7,
             label='Верхняя граница: 2·log₂(n+1)')
    plt.plot(sizes, lower_bound, 'g:', alpha=0.7,
             label='Нижняя граница: log₂(n+1)')

    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Высота дерева h(n)')
    plt.title('Красно-черное дерево: Высота от количества узлов (отсортированные ключи)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def estimate_log_constant(sizes, heights):
    log_n = [math.log2(n) for n in sizes]
    coeff = np.polyfit(log_n, heights, 1)
    c = coeff[0]  # Наклон линии
    b = coeff[1]  # Свободный член

    return c, b

def run_all_experiments():
    print("\n" + "=" * 60)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("=" * 60)

    MAX_NODES = 1000
    STEP = 50
    NUM_TRIALS = 10

    # 1. BST (случайные ключи)
    print("\n1. BST: экспериментальная зависимость высоты от количества ключей")
    print("   Условие: значения ключей не повторяются")

    sizes, bst_heights_random = run_experiment(
        BST, generate_random_values, MAX_NODES, STEP, NUM_TRIALS
    )

    plot_bst_random(sizes, bst_heights_random)

    # 2. AVL (случайные ключи)
    print("\n2. AVL на случайных ключах")
    print("   верхняя граница: ≤ 1.44·log₂(n)")
    print("   нижняя граница: log₂(n+1)")

    sizes, avl_heights_random = run_experiment(
        AVLTree, generate_random_values, MAX_NODES, STEP, NUM_TRIALS
    )

    c, b = estimate_log_constant(sizes, avl_heights_random)

    plot_avl_random(sizes, avl_heights_random)

    # 3. RB (случайные ключи)
    print("\n3. Красно-черное дерево на случайных ключах")
    print("   верхняя граница: ≤ 2.00·log₂(n+1)")
    print("   нижняя граница: log₂(n+1)")

    sizes, rb_heights_random = run_experiment(
        RBTree, generate_random_values, MAX_NODES, STEP, NUM_TRIALS
    )

    c, b = estimate_log_constant(sizes, rb_heights_random)

    plot_rb_random(sizes, rb_heights_random)

    # 4. AVL (отсортированные ключи)
    print("\n4. AVL на отсортированных ключах")
    print("   верхняя граница: ≤ 1.44·log₂(n) (гарантируется)")
    print("   нижняя граница: log₂(n)")

    sizes, avl_heights_sorted = run_experiment(
        AVLTree, generate_sorted_values, MAX_NODES, STEP, 1
    )

    c, b = estimate_log_constant(sizes, avl_heights_sorted)

    plot_avl_sorted(sizes, avl_heights_sorted)

    # 5. RB (отсортированные ключи)
    print("\n5. Красно-черное дерево на отсортированных ключах")
    print("   верхняя граница: ≤ 2.00·log₂(n+1)")
    print("   нижняя граница: log₂(n+1)")

    sizes, rb_heights_sorted = run_experiment(
        RBTree, generate_sorted_values, MAX_NODES, STEP, 1
    )

    c, b = estimate_log_constant(sizes, rb_heights_sorted)

    plot_rb_sorted(sizes, rb_heights_sorted)

# Тестирование
if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ДЕРЕВЬЕВ")
    print("=" * 60)

    print("\n=== Тестирование BST ===")
    bst = BST()
    test_values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45, 55, 65, 75, 85]
    print(f"Вставляем значения: {test_values}")
    for value in test_values:
        bst.insert(value)

    print("Обходы BST:")
    print("Прямой обход: ", end="")
    bst.pre_order_traversal()
    print("\nЦентрированный обход: ", end="")
    bst.in_order_traversal()
    print("\nОбратный обход: ", end="")
    bst.post_order_traversal()
    print("\nОбход в ширину: ", end="")
    bst.level_order_traversal()

    print(f"\n\nПоиск в BST:")
    print(f"Поиск 40: {bst.search(40)}")
    print(f"Поиск 90: {bst.search(90)}")
    print(f"Поиск 10: {bst.search(10)}")
    print(f"Минимум: {bst.find_min()}")
    print(f"Максимум: {bst.find_max()}")
    print(f"Высота BST: {bst.get_height()}")

    print("\nТестирование удаления в BST:")
    print("Удаляем 20 (лист):")
    bst.delete(20)
    print("Центрированный обход: ", end="")
    bst.in_order_traversal()

    print("\n\nУдаляем 30 (узел с одним потомком):")
    bst.delete(30)
    print("Центрированный обход: ", end="")
    bst.in_order_traversal()

    print("\n\nУдаляем 50 (узел с двумя потомками, корень):")
    bst.delete(50)
    print("Центрированный обход: ", end="")
    bst.in_order_traversal()
    print(f"\nНовый корень: {bst.root.value if bst.root else 'None'}")
    print(f"Высота после удалений: {bst.get_height()}")

    # Тестирование AVL
    print("\n" + "=" * 60)
    print("=== Тестирование AVLTree ===")
    avl = AVLTree()

    print("Вставляем значения:", test_values)
    for value in test_values:
        avl.insert(value)

    print("\nОбходы AVL:")
    print("Прямой обход: ", end="")
    avl.pre_order_traversal()
    print("\nЦентрированный обход: ", end="")
    avl.in_order_traversal()
    print("\nОбратный обход: ", end="")
    avl.post_order_traversal()
    print("\nОбход в ширину: ", end="")
    avl.level_order_traversal()

    print("\n\nПоиск в AVL:")
    print("Поиск 30:", avl.search(30))
    print("Поиск 100:", avl.search(100))
    print("Поиск 60:", avl.search(60))

    print("\nМинимум:", avl.find_min())
    print("Максимум:", avl.find_max())
    print(f"Высота AVL: {avl.get_height()}")

    print("\nУдаляем 20")
    avl.delete(20)
    print("Центрированный обход после удаления 20: ", end="")
    avl.in_order_traversal()

    print("\n\nУдаляем 50 (корень)")
    avl.delete(50)
    print("Центрированный обход после удаления 50: ", end="")
    avl.in_order_traversal()
    print(f"\nНовый корень: {avl.root.value if avl.root else 'None'}")
    print(f"Высота после удалений: {avl.get_height()}")

    print("\n" + "=" * 60)
    print("=== Тестирование RBTree ===")
    rb = RBTree()

    print("Вставляем значения:", test_values)
    for value in test_values:
        rb.insert(value)

    print("\nПосле вставки всех значений:")
    print("Центрированный обход: ", end="")
    rb.in_order_traversal()
    print("\nВысота RB дерева:", rb.get_height())

    # Тест 1
    print("\n1. Удаляем 10 (лист):")
    rb.delete(10)
    print("Центрированный обход: ", end="")
    rb.in_order_traversal()
    print(f", Высота: {rb.get_height()}")

    # Тест 2
    print("\n2. Удаляем 30 (с одним ребёнком после удаления 10):")
    rb.delete(30)
    print("Центрированный обход: ", end="")
    rb.in_order_traversal()
    print(f", Высота: {rb.get_height()}")

    # Тест 3
    print("\n3. Удаляем 40 (оба ребёнка):")
    rb.delete(40)
    print("Центрированный обход: ", end="")
    rb.in_order_traversal()
    print(f", Высота: {rb.get_height()}")

    # Тест 4
    print("\n4. Проверка поиска:")
    print("Поиск 25:", rb.search(25))
    print("Поиск 30:", rb.search(30))
    print("Поиск 60:", rb.search(60))
    print("Поиск 100:", rb.search(100))

    # Тест 5
    print("\n5. Минимум и максимум:")
    print("Минимум:", rb.find_min())
    print("Максимум:", rb.find_max())

    # Тест 6
    print("\n6. Проверка свойств RB дерева:")
    print("Цвет корня:", rb.root.color)
    print("Высота RB дерева:", rb.get_height())

    run_all_experiments()