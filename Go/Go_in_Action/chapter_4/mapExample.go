package main

import "fmt"

func declareMapWithMake() {
	dict := make(map[string]int)
	fmt.Println(dict)
}

func declareMapWithLiteral() {
	dict := map[string]string{"Red": "#da1337"}
	fmt.Println(dict)
}

func assignValuesToMap() {
	colors := map[string]string{}
	colors["Red"] = "#da1337"
}

func declareNullMap() {
	var colors map[string]string
	// A nil map can't used to store key/value pairs
	fmt.Println(colors)
}

func getValueFromKey() {
	colors := map[string]string{"Red": "#da1337"}
	value, exist := colors["Blue"]

	// version 1
	if exist {
		fmt.Println(value)
	}

	// version 2
	if value != "" {
		fmt.Println(value)
	}
}

func iterateMap() {
	colors := map[string]string{
		"AliceBlue":   "#f0f8ff",
		"Coral":       "#ff7F50",
		"DarkGray":    "#a9a9a9",
		"ForestGreen": "#228b22",
	}

	for key, value := range colors {
		fmt.Printf("Key: %s  Value: %s\n", key, value)
	}
}

func main() {
	fmt.Println("This is map example")
}
