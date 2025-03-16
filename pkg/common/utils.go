package common

import (
	"fmt"
	"os"
)

func Log(message string) {
	fmt.Println(message)
}

func PrintAndTerminate(message string) {
	Log(message)
	os.Exit(1)
}
