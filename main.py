from sys import exit
from queue import PriorityQueue
import pygame
import copy

# Default Map
MAP = [[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]]

# Tile size
TILE_X = 25
TILE_Y = 25

# Starting from 0
# Maximum Row index
ROW = len(MAP) - 1
# Maximum Column index
COLUMN = len(MAP[ROW]) - 1

# Pygame settings
WINDOW_SIZE = (1000, 775)
FPS = 60

# Representation of data in map 
ENEMY = -1
EMPTY = 0
CHARACTER = 1
WALL = 2
EXPLORED = 3
SOLUTION = 4
QUEUE = 5

# Color RGB values
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (56, 252, 72)
CYAN = (116, 252, 250)
LIGHT_GREY = (214, 214, 214)
GREY = (116, 117, 117)
DARK_GREY = (82, 78, 82)
LIGHT_GREEN = (149, 224, 119)
LIGHT_BLUE = (170, 201, 250)
DARK_BLUE = (53, 75, 94)
DARK_GREEN = (58, 163, 16)

def timer(start_time):
    """Used to display timer"""
    curr_time = pygame.time.get_ticks()*0.001 - start_time
    timer_surf = font.render(f"Timer {curr_time:.4f}s", False, "White")
    timer_rect = timer_surf.get_rect(topleft=(815, 60))
    screen.blit(timer_surf, timer_rect)
    return curr_time

def draw_map(MAP_state):
    """Used to display map in user interface"""
    for row in range(len(MAP_state)):
        for column in range(len(MAP_state[row])):
            # If the current grid is empty, display grey 
            if MAP_state[row][column] == EMPTY:
                pygame.draw.rect(
                    screen, GREY,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid is the main character, display yellow
            elif MAP_state[row][column] == CHARACTER:
                pygame.draw.rect(
                    screen, YELLOW,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid is the enemy, display red
            elif MAP_state[row][column] == ENEMY:
                pygame.draw.rect(
                    screen, RED,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid is a wall, display dark grey
            elif MAP_state[row][column] == WALL:
                pygame.draw.rect(
                    screen, DARK_GREY,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid has been explored, display light grey
            elif MAP_state[row][column] == EXPLORED:
                pygame.draw.rect(
                    screen, LIGHT_GREY,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid is one of the solution path, display green
            elif MAP_state[row][column] == SOLUTION:
                pygame.draw.rect(
                    screen, GREEN,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))
            # If the current grid is in queue, display cyan
            elif MAP_state[row][column] == QUEUE:
                pygame.draw.rect(
                    screen, CYAN,
                    pygame.Rect(column * TILE_X, row * TILE_Y, TILE_X - 1,
                                TILE_Y - 1))

def display_num_nodes(cube_explored, cube_solution):
    """Display number of explored nodes in user interface"""
    cube_explored_surf = font.render(f"Explored:\n{cube_explored} blocks", False, "White")
    cube_explored_rect = cube_explored_surf.get_rect(topleft=(815, 160))
    cube_solution_surf = font.render(f"Shortest Path:\n{cube_solution} blocks", False, "White")
    cube_solution_rect = cube_solution_surf.get_rect(topleft=(815, 230))
    screen.blit(cube_explored_surf, cube_explored_rect)
    screen.blit(cube_solution_surf, cube_solution_rect)

def reset():
    """To reset the whole map"""
    global character_button_pressed
    global enemy_button_pressed
    global wall_button_pressed
    global MAP_state
    global character_pos
    global enemy_pos
    # Reset character button to not being pressed
    character_button_pressed = 0
    # Reset enemy button to not being pressed
    enemy_button_pressed = 0
    # Reset wall button to not being pressed
    wall_button_pressed = 0
    # Clear character position
    character_pos.clear()
    # Clear enemy position
    enemy_pos.clear()
    # Clear Map by assigning it to default map
    MAP_state = copy.deepcopy(MAP)


def history_window():
    """To display map history and used algorithm"""
    history_surf = font.render(f"History", False, "White")
    history_rect = history_surf.get_rect(topleft=(815, 20))
    back_button = Button("Back", 15, (815, 737), 70, 30, screen, 3, LIGHT_BLUE, DARK_BLUE)
    history_button_list = []
  
    # Create map history button based on length of history list
    for i in range(len(history)):
        name, timer, cube_explored, cube_solution = history[i][0], history[i][1], history[i][3], history[i][4]
        history_button_list.append(
            Button(f"{name}\n{timer:.4f}s\nExplored: {cube_explored} blocks\nSolution: {cube_solution} blocks", 
                   15, (815,  60 + (i)*90), 170, 80,
                   screen, 3, LIGHT_BLUE, DARK_BLUE))
        
    history_map = MAP_state
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill("black")
        draw_map(history_map)
        screen.blit(history_surf, history_rect)

        # Display list of map history button
        for index in range(len(history_button_list)):
            if history_button_list[index].draw():
                history_map = history[index][2]

        # Back button (Return to main page)
        if back_button.draw():
            break

        pygame.display.update()
        clock.tick(FPS)


class Node:

    def __init__(self, row, column, MAP_state):
        """Initialize Node object with parameters row, column, prior, queue, visisted, and neighbours"""
        self.row = row
        self.column = column
        self.prior = None
        self.queue = False
        self.visited = False
        self.neighbours = []

    def set_neighbours(self, searching_map):
        """To determine available successors of each Node object to update neighbour parameter"""
        if self.row > 0 and MAP_state[self.row - 1][self.column] != WALL: #UP
            self.neighbours.append(searching_map[self.row - 1][self.column])

        if self.column < COLUMN and MAP_state[self.row][self.column + 1] != WALL: #RIGHT
            self.neighbours.append(searching_map[self.row][self.column + 1])

        if self.row < ROW and MAP_state[self.row + 1][self.column] != WALL: #DOWN
            self.neighbours.append(searching_map[self.row + 1][self.column])

        if self.column > 0 and MAP_state[self.row][self.column - 1] != WALL: #LEFT
            self.neighbours.append(searching_map[self.row][self.column - 1])

    def get_pos(self):
        """To get row and column of each Node"""
        return self.row, self.column


def make_searching_map(row, column, MAP_state):
    """To initialize a 2D array of Node objects"""
    searching_map = []

    for i in range(ROW + 1):
        searching_map.append([])
        for j in range(COLUMN + 1):
            node = Node(i, j, MAP_state)
            searching_map[i].append(node)

    return searching_map
  
def calc_h_score(pos1, pos2):
    """To calculate Manhattan distance from pos1 to pos2"""
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)


def BFS(MAP_state, searching_map, character_pos, enemy_pos):
    """Search solution path using BFS algorithm"""
    algorithm_surf = font.render(f"BFS", False, "White")
    algorithm_rect = algorithm_surf.get_rect(topleft=(815, 20))
    start_time = pygame.time.get_ticks() * 0.001

    # Rearrange character_pos into row, column instead of x, y
    column_char, row_char = character_pos
    column_enemy, row_enemy = enemy_pos
    enemy = searching_map[row_enemy][column_enemy]
    character = searching_map[row_char][column_char]
  
    #list of unexplored nodes
    queue = []
    queue.append(enemy)
    enemy.queue = True
    enemy.visited = True
  
    #list of nodes in solution path
    path = []
  
    #indicate existance of solution: 0 if not found, 1 if found
    flag = 0

    #to indicate status of searching: True if search should continue, false if search should end 
    searching = True

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill("black")
        screen.blit(algorithm_surf, algorithm_rect)
        curr_time = timer(start_time)
        draw_map(MAP_state)
        pygame.display.update()
        clock.tick(FPS)
        cube_explored = 0
        cube_solution = 0
      
        for i in range(ROW + 1):
            for j in range(COLUMN + 1):
                box = searching_map[i][j]

                #change visual colour of grids based on their status
                if box.queue:
                    if box != character and box != enemy:
                        MAP_state[i][j] = QUEUE
                if box.visited:
                    cube_explored = cube_explored + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = EXPLORED
                if box in path:
                    cube_solution = cube_solution + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = SOLUTION

        if len(queue) > 0 and searching:
            #FIFO appraoch, explore least recently added Node
            current_box = queue.pop(0)
            current_box.visited = True
          
            #check if targeted Node is reached
            if current_box == character:
                searching = False
                flag = 1
                while current_box is not None:
                    path.append(current_box)
                    current_box = current_box.prior
                  
            #obtain new Nodes for exploration
            else:
                for neighbour in current_box.neighbours:
                    if not neighbour.queue:
                        neighbour.queue = True
                        neighbour.prior = current_box
                        queue.append(neighbour)
        else:
            searching = False
            break
            
    # Put information about this search into history list
    history.insert(0, ["BFS", curr_time, MAP_state, cube_explored, cube_solution])
    resetting = False
    solution = "Contain solution" if flag else "No solution"
    solution_surf = font.render(f"{solution}", False, "White")
    solution_rect = solution_surf.get_rect(topleft=(815, 100))
  
    while not resetting:
        events = pygame.event.get()
      
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Display map
        draw_map(MAP_state)
        screen.blit(solution_surf, solution_rect)
        display_num_nodes(cube_explored, cube_solution)

        # If reset button is clicked, reset and return to main page
        if reset_button.draw():
            reset()
            resetting = True

        pygame.display.update()
        clock.tick(FPS)


def DFS(MAP_state, searching_map, character_pos, enemy_pos):
    """Search solution path using DFS algorithm"""
    algorithm_surf = font.render(f"DFS", False, "White")
    algorithm_rect = algorithm_surf.get_rect(topleft=(815, 20))
    start_time = pygame.time.get_ticks() * 0.001
  
    column_char, row_char = character_pos
    column_enemy, row_enemy = enemy_pos
    character = searching_map[row_char][column_char]
    enemy = searching_map[row_enemy][column_enemy]
    
    queue = []
    queue.append(enemy)
    enemy.queue = True
    enemy.visited = True
  
    path = []
    flag = 0
    searching = True
    
    while True:
        events = pygame.event.get()    
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill("black")
        screen.blit(algorithm_surf, algorithm_rect)
        curr_time = timer(start_time)
        draw_map(MAP_state)
        pygame.display.update()
        clock.tick(FPS)
        cube_explored = 0
        cube_solution = 0
      
        for i in range(ROW + 1):
            for j in range(COLUMN + 1):
                box = searching_map[i][j]

                if box.queue:
                    if box != character and box != enemy:
                        MAP_state[i][j] = QUEUE
                if box.visited:
                    cube_explored = cube_explored + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = EXPLORED
                if box in path:
                    cube_solution = cube_solution + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = SOLUTION

        if len(queue) > 0 and searching:
            #LIFO approach, explore most recently added Node 
            current_box = queue.pop(-1)
            current_box.visited = True
          
            if current_box == character:
                searching = False
                flag = 1
                while current_box is not None:
                    path.append(current_box)
                    current_box = current_box.prior

            else:
                for neighbour in current_box.neighbours:
                    if not neighbour.queue:
                        neighbour.queue = True
                        neighbour.prior = current_box
                        queue.append(neighbour)
        else:
            searching = False
            break

    history.insert(0, ["DFS", curr_time, MAP_state, cube_explored, cube_solution])
    resetting = False
    solution = "Contain solution" if flag else "No solution"
    solution_surf = font.render(f"{solution}", False, "White")
    solution_rect = solution_surf.get_rect(topleft=(815, 100))
  
    while not resetting:
        events = pygame.event.get()
      
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        draw_map(MAP_state)
        screen.blit(solution_surf, solution_rect)
        display_num_nodes(cube_explored, cube_solution)

        if reset_button.draw():
            reset()
            resetting = True

        pygame.display.update()
        clock.tick(FPS)


def A_star(MAP_state, searching_map, character_pos, enemy_pos, calc_h_score):
    """Search solution path using A star algorithm"""
  
    algorithm_surf = font.render(f"A*", False, "White")
    algorithm_rect = algorithm_surf.get_rect(topleft=(815, 20))
    start_time = pygame.time.get_ticks() * 0.001
  
    column_char, row_char = character_pos
    character = searching_map[row_char][column_char]
    column_enemy, row_enemy = enemy_pos
    enemy = searching_map[row_enemy][column_enemy]
  

    # Priority queue data structure in the form of (first priority, second priority, data)
    # dequeue node with least f_score; for nodes with the same f_score, dequeue least recently added node
    count = 0
    queue = PriorityQueue()
    enemy.queue = True
    enemy.visited = True
    
    path = []
    flag = 0
    
    searching = True

    # Initialize path cost and total function score of every node with infinity, to replace later
    g_score = {node: float("inf") for row in searching_map for node in row}
    f_score = {node: float("inf") for row in searching_map for node in row}

    # Calculate f_score of enemy node (start node)
    g_score[enemy] = 0
    f_score[enemy] = g_score[enemy] + calc_h_score(enemy.get_pos(), character.get_pos())
    queue.put((f_score[enemy], count, enemy))

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill("black")
        screen.blit(algorithm_surf, algorithm_rect)
        curr_time = timer(start_time)
        draw_map(MAP_state)
        pygame.display.update()
        clock.tick(FPS)
        cube_explored = 0
        cube_solution = 0
      
        for i in range(ROW + 1):
            for j in range(COLUMN + 1):
                box = searching_map[i][j]

                if box.queue:
                    if box != character and box != enemy:
                        MAP_state[i][j] = QUEUE
                if box.visited:
                    cube_explored = cube_explored + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = EXPLORED
                if box in path:
                    cube_solution = cube_solution + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = SOLUTION
                        
        if searching and not queue.empty():
            # Obtain node from tuple in priority queue for exploring
            current_box = queue.get()[2]
            current_box.visited = True
            
            if current_box == character:
                searching = False
                flag = 1
                while current_box is not None:
                    path.append(current_box)
                    current_box = current_box.prior

            else:
                for neighbour in current_box.neighbours:
                    temp_g_score = g_score[current_box] + 1

                    # To find best path to each neighbour
                    if temp_g_score < g_score[neighbour]:
                        neighbour.prior = current_box
                        g_score[neighbour] = temp_g_score
                        f_score[neighbour] = temp_g_score + calc_h_score(neighbour.get_pos(), character.get_pos())
                        
                    if not neighbour.queue:
                        count += 1
                        neighbour.queue = True
                        queue.put((f_score[neighbour], count, neighbour))
        else:
            searching = False
            break
            

    history.insert(0, ["A*", curr_time, MAP_state, cube_explored, cube_solution])
    resetting = False
    solution = "Contain solution" if flag else "No solution"
    solution_surf = font.render(f"{solution}", False, "White")
    solution_rect = solution_surf.get_rect(topleft=(815, 100))
  
    while not resetting:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        draw_map(MAP_state)
        screen.blit(solution_surf, solution_rect)
        display_num_nodes(cube_explored, cube_solution)

        if reset_button.draw():
            reset()
            resetting = True

        pygame.display.update()
        clock.tick(FPS)


def Greedy_BFS(MAP_state, searching_map, character_pos, enemy_pos, h_score):
    """Search solution path using Greedy Best First Search algorithm"""
    algorithm_surf = font.render(f"Greedy BFS", False, "White")
    algorithm_rect = algorithm_surf.get_rect(topleft=(815, 20))
    start_time = pygame.time.get_ticks() * 0.001
  
    column_char, row_char = character_pos
    column_enemy, row_enemy = enemy_pos
    enemy = searching_map[row_enemy][column_enemy]
    character = searching_map[row_char][column_char]

    count = 0
    queue = PriorityQueue()
    enemy.queue = True
    enemy.visited = True
    
    path = []
    flag = 0
    
    searching = True

    #only heuristic function is considered in greedy best first search
    h_score = {node: float("inf") for row in searching_map for node in row}
    h_score[enemy] = calc_h_score(enemy.get_pos(), character.get_pos())
    queue.put((h_score[enemy], count, enemy))
  
    searching = True

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill("black")
        screen.blit(algorithm_surf, algorithm_rect)
        curr_time = timer(start_time)
        draw_map(MAP_state)
        pygame.display.update()
        clock.tick(FPS)
        cube_explored = 0
        cube_solution = 0
      
        for i in range(ROW + 1):
            for j in range(COLUMN + 1):
                box = searching_map[i][j]

                if box.queue:
                    if box != character and box != enemy:
                        MAP_state[i][j] = QUEUE
                if box.visited:
                    cube_explored = cube_explored + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = EXPLORED
                if box in path:
                    cube_solution = cube_solution + 1
                    if box != character and box != enemy:
                        MAP_state[i][j] = SOLUTION

        if not queue.empty() and searching:
            current_box = queue.get()[2]
            current_box.visited = True
          
            if current_box == character:
                searching = False
                flag = 1
                while current_box is not None:
                    path.append(current_box)
                    current_box = current_box.prior

            else:
                for neighbour in current_box.neighbours:
                    if not neighbour.queue:
                        count += 1
                        neighbour.queue = True
                        neighbour.prior = current_box
                        h_score[neighbour] = calc_h_score(neighbour.get_pos(), character.get_pos())
                        queue.put((h_score[neighbour], count, neighbour))
        else:
            searching = False
            break

    history.insert(0, ["Greedy BFS", curr_time, MAP_state, cube_explored, cube_solution])
    resetting = False
    solution = "Contain solution" if flag else "No solution"
    solution_surf = font.render(f"{solution}", False, "White")
    solution_rect = solution_surf.get_rect(topleft=(815, 100))
  
    while not resetting:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        draw_map(MAP_state)
        screen.blit(solution_surf, solution_rect)
        display_num_nodes(cube_explored, cube_solution)

        if reset_button.draw():
            reset()
            resetting = True

        pygame.display.update()
        clock.tick(FPS)


class Button:
    """
    Creates a functional button. 
    Will return True or False every time the draw method is called
    depending on whether the button is being pressed
    """

    def __init__(self, text, text_size, pos, width, height, screen, elevation,
                 top_color, bot_color):
        """Initalize class Button with individual parameters"""
        # Screen to display the button
        self.screen = screen

        # Variables needed for the button
        self.pressed = False
        self.hovered = False
        self.elevation = elevation
        self.dynamic_elevation = elevation
        self.ori_y = pos[1]

        # The button itself
        self.top_rect = pygame.Rect(pos, (width, height))
        self.top_color = top_color
        self.top_rect_color = self.top_color

        # The "Shadow" of the button
        self.bottom_rect = pygame.Rect(pos, (width, height))
        self.bottom_rect_color = bot_color

        # Text on the button
        self.font = pygame.font.SysFont('Arial', text_size)
        self.text_surf = self.font.render(text, False, "Black")
        self.text_rect = self.text_surf.get_rect(center=self.top_rect.center)

    def draw(self):
        # Code for the elevation effect of the button
        self.top_rect.y = self.ori_y - self.dynamic_elevation
        self.text_rect.center = self.top_rect.center
        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elevation

        # Draw the button
        pygame.draw.rect(self.screen,
                         self.bottom_rect_color,
                         self.bottom_rect,
                         border_radius=30)
        pygame.draw.rect(self.screen,
                         self.top_rect_color,
                         self.top_rect,
                         border_radius=30)

        # Displaying the text
        self.screen.blit(self.text_surf, self.text_rect)
        return self.check_clicked()

    def check_clicked(self):
        mouse_pos = pygame.mouse.get_pos()

        # If the cursor is right on top of the button
        if self.top_rect.collidepoint(mouse_pos):
            self.top_rect_color = "Red"
            if not self.hovered:
                self.hovered = True

            # If we are holding the left click
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elevation = 0
                self.pressed = True
            else:

                # If we are not holding the left click
                self.dynamic_elevation = self.elevation

                # If we did press the button
                if self.pressed:
                    self.pressed = False
                    return True

                # If we didn't press the button
                else:
                    return False
        else:
            # If the cursor is not on top of the button
            self.hovered = False
            self.dynamic_elevation = self.elevation
            self.top_rect_color = self.top_color
            self.pressed = False
            return False

# Map state
MAP_state = copy.deepcopy(MAP)

# Start pygame engine
pygame.init()
pygame.display.set_caption('Pathfinder')
# Set screen size
screen = pygame.display.set_mode(WINDOW_SIZE)
# Set clock to manage FPS
clock = pygame.time.Clock()

# Create character button, button state and list to store character position
character_button = Button("Character", 15, (815, 60), 70, 30, screen, 3,
                          LIGHT_BLUE, DARK_BLUE)
character_button_pressed = 0
character_pos = []

# Create enemy button, button state and list to store character position
enemy_button = Button("Enemy", 15, (815, 100), 70, 30, screen, 3, LIGHT_BLUE,
                      DARK_BLUE)
enemy_button_pressed = 0
enemy_pos = []

# Create wall button and button state 
wall_button = Button("Wall", 15, (815, 140), 70, 30, screen, 3, LIGHT_BLUE,
                     DARK_BLUE)
wall_button_pressed = 0

# Create reset button
reset_button = Button("Reset", 15, (815, 737), 70, 30, screen, 3, LIGHT_BLUE,
                      DARK_BLUE)

# Create history button and history list
history_button = Button("History", 15, (815, 697), 70, 30, screen, 3,
                        LIGHT_BLUE, DARK_BLUE)
history = []

# Create copy button and copied item (initially set to None)
copy_button = Button("Copy", 15, (815, 617), 70, 30, screen, 3, LIGHT_BLUE,
                     DARK_BLUE)
copied = None

# Create paste button
paste_button = Button("Paste", 15, (815, 657), 70, 30, screen, 3, LIGHT_BLUE,
                      DARK_BLUE)

# Create BFS button
BFS_button = Button("Breadth First Search", 15, (815, 200), 170, 30, screen, 3, LIGHT_GREEN,
                    DARK_GREEN)

# Create DFS button
DFS_button = Button("Depth First Search", 15, (815, 240), 170, 30, screen, 3, LIGHT_GREEN,
                    DARK_GREEN)

# Create A star button
A_star_button = Button("A* Search", 15, (815, 280), 170, 30, screen, 3, LIGHT_GREEN,
                       DARK_GREEN)

# Create Dijikstra button
Greedy_BFS_button = Button("Greedy Best First Search", 15, (815, 320), 170, 30, screen, 3,
                         LIGHT_GREEN, DARK_GREEN)

# Font to be used
font = pygame.font.SysFont("Consolas", 19)

# Creating a text surface that will display what type of mode is being used currently
mode = ""
mode_surf = font.render(f"Mode: {mode}", False, "White")
mode_rect = mode_surf.get_rect(topleft=(815, 20))

# Main page GUI
while True:
    # Get all user events (keyboard pressed, mouse clicked, etc.)
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Creating the screen (Map, mode)
    screen.fill("black")
    mode_surf = font.render(f"Mode: {mode}", False, "White")
    screen.blit(mode_surf, mode_rect)
    draw_map(MAP_state)

    # Ensuring that history list will at most contain 6 map history
    if len(history) > 6:
        history.pop()

    # If character button is pressed
    if character_button.draw():
        # Change character button state
        character_button_pressed = 1 - character_button_pressed
        # Reset other button state
        enemy_button_pressed = 0
        wall_button_pressed = 0

    # If enemy button is pressed
    if enemy_button.draw():
        # Change enemy button state
        enemy_button_pressed = 1 - enemy_button_pressed
        # Reset other button state
        character_button_pressed = 0
        wall_button_pressed = 0

    # If wall button is pressed
    if wall_button.draw():
        # Change wall button state
        wall_button_pressed = 1 - wall_button_pressed
        # Reset other button state
        character_button_pressed = 0
        enemy_button_pressed = 0

    # If reset button is being pressed
    if reset_button.draw():
        reset()

    # If character button has been pressed
    if character_button_pressed:
        mode = "Character"
        for event in events:
            # If mouse button has been pressed
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If it is a left click
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    pos_x, pos_y = mouse_pos
                    pos_x //= TILE_X
                    pos_y //= TILE_Y
                    mouse_pos = pos_x, pos_y
                    # If mouse position is within the map
                    if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                        # If the grid is empty
                        if MAP_state[pos_y][pos_x] == EMPTY:
                            # Set grid state from empty to character
                            MAP_state[pos_y][pos_x] = CHARACTER
                            # If character position had been set before
                            if len(character_pos):
                                # Remove previous character position
                                pos_x, pos_y = character_pos.pop()
                                # Set previous character position grid state to empty
                                MAP_state[pos_y][pos_x] = EMPTY
                            # Add new character position into list
                            character_pos.append(mouse_pos)
                # If it is a right click
                elif event.button == 3:
                    mouse_pos = pygame.mouse.get_pos()
                    pos_x, pos_y = mouse_pos
                    pos_x //= TILE_X
                    pos_y //= TILE_Y
                    mouse_pos = pos_x, pos_y
                    # If mouse position is within the map
                    if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                        # If the grid is a character 
                        if MAP_state[pos_y][pos_x] == CHARACTER:
                            # Change grid state to empty
                            MAP_state[pos_y][pos_x] = EMPTY
                            # Remove charaacter position
                            character_pos.pop()

    # If enemy button has been pressed
    elif enemy_button_pressed:
        mode = "Enemy"
        for event in events:
            # If mouse button has been pressed
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If it is a left click
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    pos_x, pos_y = mouse_pos
                    pos_x //= TILE_X
                    pos_y //= TILE_Y
                    mouse_pos = pos_x, pos_y
                    # If mouse position is within the map
                    if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                        # If the grid is empty
                        if MAP_state[pos_y][pos_x] == EMPTY:
                            # Set grid state from empty to enemy
                            MAP_state[pos_y][pos_x] = ENEMY
                            # If enemy position had been set before
                            if len(enemy_pos):
                                # Remove previous enemy position
                                pos_x, pos_y = enemy_pos.pop()
                                # Set previous enemy position grid state to empty
                                MAP_state[pos_y][pos_x] = EMPTY
                            # Add new enemy position into list
                            enemy_pos.append(mouse_pos)
                # If it is a right click
                elif event.button == 3:
                    mouse_pos = pygame.mouse.get_pos()
                    pos_x, pos_y = mouse_pos
                    pos_x //= TILE_X
                    pos_y //= TILE_Y
                    mouse_pos = pos_x, pos_y
                    # If mouse position is within the map
                    if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                        # If the grid is an enemy
                        if MAP_state[pos_y][pos_x] == ENEMY:
                            # Change grid state to empty
                            MAP_state[pos_y][pos_x] = EMPTY
                            # Remove enemy position
                            enemy_pos.pop()

    # If wall button has been pressed
    elif wall_button_pressed:
        mode = "Wall"
        # If left click is being HELD
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            pos_x, pos_y = mouse_pos
            pos_x //= TILE_X
            pos_y //= TILE_Y
            mouse_pos = pos_x, pos_y
            # If mouse position is within the map
            if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                # If the grid is empty
                if MAP_state[pos_y][pos_x] == EMPTY:
                    # Set grid state from empty to wall
                    MAP_state[pos_y][pos_x] = WALL
        elif pygame.mouse.get_pressed()[2]:
            mouse_pos = pygame.mouse.get_pos()
            pos_x, pos_y = mouse_pos
            pos_x //= TILE_X
            pos_y //= TILE_Y
            mouse_pos = pos_x, pos_y
            # If mouse position is within the map
            if 0 < pos_x < COLUMN and 0 < pos_y < ROW:
                # If the grid is empty
                if MAP_state[pos_y][pos_x] == WALL:
                    # Set grid state from empty to wall
                    MAP_state[pos_y][pos_x] = EMPTY


    # If no button has been pressed
    else:
        mode = ""
    
    # If character position has been set
    if character_pos and enemy_pos:
        searching_map = make_searching_map(ROW, COLUMN, MAP_state)
        for row in searching_map:
            for node in row:
                node.set_neighbours(searching_map)

        # If BFS button is pressed
        if BFS_button.draw():
            BFS(MAP_state, searching_map, character_pos[0], enemy_pos[0])
        
        # If DFS button is pressed
        elif DFS_button.draw():
            DFS(MAP_state, searching_map, character_pos[0], enemy_pos[0])
        
        # If A star button is pressed
        elif A_star_button.draw():
            A_star(MAP_state, searching_map, character_pos[0], enemy_pos[0], calc_h_score)

        # If Dijikstra button is pressed
        elif Greedy_BFS_button.draw():
            Greedy_BFS(MAP_state, searching_map, character_pos[0], enemy_pos[0], calc_h_score)

    # If copy button is pressed
    if copy_button.draw():
        copied = [
            copy.deepcopy(MAP_state),
            copy.deepcopy(character_pos),
            copy.deepcopy(enemy_pos)
        ]

    # If there is a copied item, display paste button
    if copied is not None:
        # If paste button is pressed
        if paste_button.draw():
            reset()
            MAP_state = copy.deepcopy(copied[0])
            # If there is a character position
            if len(copied[1]):
                character_pos.append(copied[1][0])
            # If there is an enemy position
            if len(copied[2]):
                enemy_pos.append(copied[2][0])

    # If there is map history (If history list is not empty)
    if history:
        if history_button.draw():
            reset()
            history_window()

    pygame.display.update()
    clock.tick(FPS)