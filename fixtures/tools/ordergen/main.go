// Command ordergen dumps GoBot's authoritative move-enumeration order for every
// record of fixtures/gobot_search_parity.jsonl, so the Rust port can be checked
// against it byte-for-byte.
//
// It deliberately uses the *real* game package (game.FromSnapshot +
// game.NewPosition + ForEachSearchAction / LegalActions), so the golden file is
// the Go engine's own output, not a re-derivation.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"virusgame/game"
)

type jsonCell struct {
	Owner int    `json:"owner"`
	Kind  string `json:"kind"`
}

type jsonPos struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

type jsonAction struct {
	Type   string  `json:"type"`
	Target jsonPos `json:"target"`
	Pos1   jsonPos `json:"pos1"`
	Pos2   jsonPos `json:"pos2"`
}

type record struct {
	Board       [][]jsonCell `json:"board"`
	Player      int          `json:"player"`
	MovesLeft   int          `json:"movesLeft"`
	NeutralUsed []bool       `json:"neutralUsed"`
	Action      jsonAction   `json:"action"`
}

// Actions are emitted compactly: ["M", row, col] for a move,
// ["N", row1, col1, row2, col2] for a neutral placement. Order is the order
// ForEachSearchAction yielded them.
type outRecord struct {
	Index         int             `json:"index"`
	LegalCount    int             `json:"legalCount"`
	Strategic     bool            `json:"strategic"`
	StateHash     uint64          `json:"stateHash"`
	AfterHash     uint64          `json:"afterHash"`
	SearchActions [][]interface{} `json:"searchActions"`
}

// stateHash transcribes search.stateHash (unexported) from the public State
// API. The Java port carries the same transcription and passes parity, so this
// is the reference the Rust State::state_hash must reproduce.
func stateHash(state game.State) uint64 {
	const prime = uint64(1099511628211)
	hash := uint64(1469598103934665603)
	add := func(value byte) {
		hash ^= uint64(value)
		hash *= prime
	}
	add(byte(state.Rows()))
	add(byte(state.Cols()))
	add(byte(state.CurrentPlayer()))
	add(byte(state.MovesLeft()))
	for player := game.Player(1); player <= 4; player++ {
		if state.Active(player) {
			add(byte(player) | 0x10)
		}
		if state.NeutralUsed(player) {
			add(byte(player) | 0x20)
		}
	}
	for row := 0; row < state.Rows(); row++ {
		for col := 0; col < state.Cols(); col++ {
			cell, _ := state.At(game.Pos{Row: row, Col: col})
			add(byte(cell.Owner)<<3 | byte(cell.Kind))
		}
	}
	return hash
}

var kinds = map[string]game.CellKind{
	"EMPTY":     game.Empty,
	"NORMAL":    game.Normal,
	"BASE":      game.Base,
	"FORTIFIED": game.Fortified,
	"NEUTRAL":   game.Neutral,
}

func main() {
	in := flag.String("in", "", "input parity jsonl")
	out := flag.String("out", "", "output golden jsonl")
	flag.Parse()

	source, err := os.Open(*in)
	if err != nil {
		panic(err)
	}
	defer source.Close()

	sink, err := os.Create(*out)
	if err != nil {
		panic(err)
	}
	defer sink.Close()
	writer := bufio.NewWriter(sink)
	defer writer.Flush()

	scanner := bufio.NewScanner(source)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	index := 0
	for scanner.Scan() {
		var rec record
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			panic(fmt.Sprintf("line %d: %v", index+1, err))
		}
		rows := len(rec.Board)
		cols := len(rec.Board[0])

		board := make([][]game.Cell, rows)
		for r := 0; r < rows; r++ {
			board[r] = make([]game.Cell, cols)
			for c := 0; c < cols; c++ {
				kind, ok := kinds[rec.Board[r][c].Kind]
				if !ok {
					panic("bad kind " + rec.Board[r][c].Kind)
				}
				board[r][c] = game.Cell{Owner: game.Player(rec.Board[r][c].Owner), Kind: kind}
			}
		}
		bases := []game.Pos{{Row: 0, Col: 0}, {Row: rows - 1, Col: cols - 1}}
		active := make([]bool, 2)
		for p := 0; p < 2; p++ {
			cell := board[bases[p].Row][bases[p].Col]
			active[p] = cell.Owner == game.Player(p+1) && cell.Kind == game.Base
		}
		neutralUsed := make([]bool, 2)
		copy(neutralUsed, rec.NeutralUsed)

		snapshot := game.Snapshot{
			Rows: rows, Cols: cols, Board: board, Bases: bases,
			Active: active, NeutralUsed: neutralUsed,
			Current: game.Player(rec.Player), MovesLeft: rec.MovesLeft,
		}
		state, err := game.FromSnapshot(snapshot)
		if err != nil {
			panic(fmt.Sprintf("record %d: FromSnapshot: %v", index, err))
		}
		position := game.NewPosition(state)

		legal := state.LegalActions()
		moveCount := 0
		for _, a := range legal {
			if a.Kind == game.Move {
				moveCount++
			}
		}
		ownedNormals := 0
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				cell := board[r][c]
				if cell.Owner == game.Player(rec.Player) && cell.Kind == game.Normal {
					ownedNormals++
				}
			}
		}
		canPlace := rec.MovesLeft == 3 && !neutralUsed[rec.Player-1]
		var recorded game.Action
		if rec.Action.Type == "MOVE" {
			recorded = game.Action{Kind: game.Move,
				Target: game.Pos{Row: rec.Action.Target.Row, Col: rec.Action.Target.Col}}
		} else {
			recorded = game.Action{Kind: game.PlaceNeutrals, Neutrals: [2]game.Pos{
				{Row: rec.Action.Pos1.Row, Col: rec.Action.Pos1.Col},
				{Row: rec.Action.Pos2.Row, Col: rec.Action.Pos2.Col},
			}}
		}
		after, err := state.Apply(recorded)
		if err != nil {
			panic(fmt.Sprintf("record %d: Apply(recorded): %v", index, err))
		}

		result := outRecord{
			Index:      index,
			LegalCount: len(legal),
			Strategic:  canPlace && moveCount+ownedNormals*(ownedNormals-1)/2 > 32,
			StateHash:  stateHash(state),
			AfterHash:  stateHash(after),
		}
		position.ForEachSearchAction(func(a game.Action) bool {
			if a.Kind == game.Move {
				result.SearchActions = append(result.SearchActions,
					[]interface{}{"M", a.Target.Row, a.Target.Col})
			} else {
				result.SearchActions = append(result.SearchActions,
					[]interface{}{"N", a.Neutrals[0].Row, a.Neutrals[0].Col,
						a.Neutrals[1].Row, a.Neutrals[1].Col})
			}
			return true
		})
		encoded, err := json.Marshal(result)
		if err != nil {
			panic(err)
		}
		writer.Write(encoded)
		writer.WriteByte('\n')
		index++
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "wrote %d records\n", index)
}
