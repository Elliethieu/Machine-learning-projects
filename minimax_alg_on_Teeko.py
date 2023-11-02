import copy
import random
import numpy as np

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ(self, state, current_player_piece):
        """TO DO: find all the successors of a given state"""
        #state is list of lists


        succ_states = []
        state_np = np.array(state)
        piece_count= len(np.argwhere(state_np != ' '))

        if piece_count < 8:
            empty_coords= np.argwhere(state_np == ' ')
            for empty_coord in empty_coords:
                state_copy = state_np.copy(order='C')
                state_copy[empty_coord[0]][empty_coord[1]] = current_player_piece
                succ_states.append(state_copy)


        else:         # 2) find successor states when drop_phase = False
            coordinate_of_pieces = np.argwhere(state_np == current_player_piece) # a) find the pieces coordinates:

            #for each piece, find the legal surrounding moves:
            for piece_coord in coordinate_of_pieces:
                up = piece_coord - np.array((1, 0))
                up_left = piece_coord - np.array((1, 1))
                up_right= piece_coord + np.array((-1, 1))
                down = piece_coord + np.array((1, 0))
                down_left= piece_coord + np.array((1, -1))
                down_right= piece_coord + np.array((1, 1))
                left = piece_coord - np.array((0, 1))
                right = piece_coord + np.array((0, 1))
                surrounding_coords = [up,up_left, up_right, down,down_left,down_right, left, right]

                for surr_coord in surrounding_coords:
                    if 0 <= surr_coord[0] <= 4 and 0 <= surr_coord[1] <= 4 and state_np[surr_coord[0]][surr_coord[1]] == ' ':
                        state_copy = state_np.copy(order='C')
                        state_copy[piece_coord[0], piece_coord[1]] = state_np[surr_coord[0], surr_coord[1]]
                        state_copy[surr_coord[0], surr_coord[1]] = state_np[piece_coord[0], piece_coord[1]]

                        succ_state= state_copy

                        succ_states.append(succ_state)

        return succ_states

    def piecesPosition(self,state):
        b = []
        r = []
        for row in range(5):
            for col in range(5):
                if state[row][col] == 'b':
                    b.append((row,col))
                elif state[row][col] == 'r':
                    r.append((row,col))
        return b,r

    def heuristic_game_value(self, state, piece): # check largest number of pieces connected
        b,r = self.piecesPosition(state)
        if piece == 'b':
            mine = 'b'
            oppo = 'r'

        elif piece == 'r':
            mine = 'r'
            oppo = 'b'

        # for horizontal
        mymax = 0
        oppmax = 0
        my_count = 0
        opp_count = 0

        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == mine:
                    my_count += 1
            if my_count > mymax:
                mymax = my_count
            my_count = 0
        i=0
        j=0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == oppo:
                    opp_count += 1
            if opp_count > oppmax:
                oppmax = opp_count
            opp_count = 0

        # for vertical
        for i in range(len(state)):
            for j in range(len(state)):
                if state[j][i] == mine:
                    my_count += 1
            if my_count > mymax:
                mymax = my_count
            my_count = 0
        i=0
        j=0
        for i in range(len(state)):
            for j in range(len(state)):
                if state[j][i] == oppo:
                    opp_count += 1
            if opp_count > oppmax:
                oppmax = opp_count
            opp_count = 0


        # for / diagonal
        my_count = 0
        opp_count = 0

        for row in range(3, 5):
            for i in range(2):
                if state[row][i] == mine:
                    my_count += 1
                if state[row - 1][i + 1] == mine:
                    my_count += 1
                if state[row - 2][i + 2] == mine:
                    my_count += 1
                if state[row - 3][i + 3] == mine:
                    my_count += 1

                if my_count > mymax:
                    mymax = my_count
                my_count = 0

        row = 0
        i= 0

        for row in range(3, 5):
            for i in range(2):
                if state[row][i] == oppo:
                    opp_count += 1
                if state[row - 1][i + 1] == oppo:
                    opp_count += 1
                if state[row - 2][i + 2] == oppo:
                    opp_count += 1
                if state[row - 3][i + 3] == oppo:
                    opp_count += 1
                if opp_count > oppmax:
                    oppmax = opp_count
                opp_count = 0

        # for \ diagonal
        my_count = 0
        opp_count = 0
        row = 0
        i = 0
        for row in range(2):
            for i in range(2):
                if state[row][i] == mine:
                    my_count += 1
                if state[row + 1][i + 1] == mine:
                    my_count += 1
                if state[row + 2][i + 2] == mine:
                    my_count += 1
                if state[row + 3][i + 3] == mine:
                    my_count += 1
                if my_count > mymax:
                    mymax = my_count
                my_count = 0

        row = 0
        i = 0
        for row in range(2):
            for i in range(2):
                if state[row][i] == oppo:
                    opp_count += 1
                if state[row + 1][i + 1] == oppo:
                    opp_count += 1
                if state[row + 2][i + 2] == oppo:
                    opp_count += 1
                if state[row + 3][i + 3] == oppo:
                    opp_count += 1
                if opp_count > oppmax:
                    oppmax = opp_count
                opp_count = 0

        # for 2X2
        my_count = 0
        opp_count = 0
        row = 0
        i = 0
        for row in range(4):
            for i in range(4):
                if state[row][i] == mine:
                    my_count += 1
                if state[row][i + 1] == mine:
                    my_count += 1
                if state[row + 1][i] == mine:
                    my_count += 1
                if state[row + 1][i + 1]== mine:
                    my_count += 1
                if my_count > mymax:
                    mymax = my_count
                my_count = 0

        row = 0
        i = 0
        for row in range(4):
            for i in range(4):
                if state[row][i] == oppo:
                    opp_count += 1
                if state[row][i + 1] == oppo:
                    opp_count += 1
                if state[row + 1][i] == oppo:
                    opp_count += 1
                if state[row + 1][i + 1]== oppo:
                    opp_count += 1
                if opp_count > oppmax:
                    oppmax = opp_count
                opp_count = 0

        if mymax == oppmax:
            return 0, state
        if mymax >= oppmax:
            return mymax/6, state # if mine is longer than opponent, return positive float

        return (-1) * oppmax/6, state # if opponent is longer than mine, return negative float

    def Max_Value(self, state, depth, alpha= float('-Inf'), beta = float('Inf')):
        bstate = state
        depth_limit = 4
        if self.game_value(state) != 0:
            return self.game_value(state), state

        if depth >= depth_limit:
           return self.heuristic_game_value(state, self.my_piece)

        else:
            #alpha= float('-Inf')
            for successor in self.succ(state, self.my_piece):
                succ_value, state_returned = self.Min_Value(successor, depth + 1, alpha, beta)
                if succ_value > alpha:
                    alpha = succ_value #update alpha
                    bstate= successor #update succ
                if alpha >= beta:
                    break

        return alpha, bstate

    def Min_Value(self, state, depth, alpha = float('-Inf'), beta= float('Inf')):
        bstate = state
        depth_limit = 3
        if self.game_value(state) != 0:
            return self.game_value(state), state

        if depth >= depth_limit:
           return self.heuristic_game_value(state, self.my_piece)

        else:
            beta= float('Inf')
            for successor in self.succ(state, self.my_piece):
                succ_value, state_returned = self.Max_Value(successor, depth + 1, alpha, beta)
                if succ_value < beta:
                    beta = succ_value
                    bstate= successor
                if alpha >= beta:
                    break

        return beta, bstate



    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is the AI's turn.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True   # TODO: detect drop phase
        num_red = 0
        num_black = 0
        for line in state:
            num_red += len([i for i in line if i == 'r'])
            num_black += len([i for i in line if i == 'b'])

        if num_red + num_black >= 8:
            drop_phase = False



        if drop_phase == False:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will crash the code after the drop phase!
            state_copy= copy.deepcopy(state)
            value, bstate = self. Max_Value(state_copy, 0)
            state_np = np.array(state_copy)
            bstate_np = np.array(bstate)
            state_diff = np.argwhere(bstate_np != state_np)
            first_coord= state_diff[0]
            second_coord= state_diff[1]

            if bstate_np[first_coord[0]][first_coord[1]] == ' ':
                source=(int(first_coord[0]), int(first_coord[1]))
                move_to= (int(second_coord[0]), int(second_coord[1]))
                move=[move_to, source]
            else:
                move_to = (int(first_coord[0]), int(first_coord[1]))
                source = (int(second_coord[0]), int(second_coord[1]))
                move = [move_to, source]
            return move



        if drop_phase== True:
            # select an unoccupied space randomly
            #this is what the AI is doing during drop_phase, which is playing randomly
            # TODO: implement a minimax algorithm to play better
            '''
            (row, col) = (random.randint(0,4), random.randint(0,4))
            while not state[row][col] == ' ':
                (row, col) = (random.randint(0,4), random.randint(0,4))

            # ensure the destination (row,col) tuple is at the beginning of the move list'''
            move = []
            state_copy = copy.deepcopy(state)
            value, bstate = self.Max_Value(state_copy, 0)
            state_np = np.array(state_copy)
            bstate_np = np.array(bstate)
            state_diff = np.argwhere(bstate_np != state_np)
            new_drop = (int(state_diff[0][0]), int(state_diff[0][1]))
            move.insert(0, new_drop)
            return move
            #return a destination [(row, col)]

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece of your color there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ': #can only move to an empty place
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col]==self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(3,5):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row-1][col+1] == state[row-2][col+2] == state[row-3][col+3]:
                    return 1 if state[row][col]==self.my_piece else -1
        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row][col + 1] == state[row+1][col] == \
                        state[row + 1][col + 1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()