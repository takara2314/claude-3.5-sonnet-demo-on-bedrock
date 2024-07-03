package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// ClaudeRequest はClaudeモデルへのリクエスト構造体
type ClaudeRequest struct {
	AnthropicVersion string                 `json:"anthropic_version"`
	MaxTokens        int                    `json:"max_tokens"`
	System           string                 `json:"system"`
	Messages         []ClaudeRequestMessage `json:"messages"`
}

// ClaudeRequestMessage はClaudeリクエストのメッセージ構造体
type ClaudeRequestMessage struct {
	Role    string              `json:"role"`
	Content []ClaudeTextContent `json:"content"`
}

// ClaudeTextContent はテキストコンテンツの構造体
type ClaudeTextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ClaudeUsage はトークン使用量の構造体
type ClaudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ClaudeResponse はClaudeモデルからのレスポンス構造体
type ClaudeResponse struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Content      []ClaudeTextContent `json:"content"`
	StopReason   string              `json:"stop_reason"`
	StopSequence string              `json:"stop_sequence"`
	Usage        ClaudeUsage         `json:"usage"`
}

func main() {
	// AWSの設定を読み込む
	region := "us-east-1"
	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatalf("AWSの設定読み込みに失敗しました: %v", err)
	}

	// Bedrockクライアントを作成
	client := bedrockruntime.NewFromConfig(cfg)

	// Claudeモデルの設定
	modelID := "anthropic.claude-3-5-sonnet-20240620-v1:0"

	// リクエストの作成
	request := ClaudeRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        1024,
		System:           "幼稚園児を演じてください。",
		Messages: []ClaudeRequestMessage{
			{
				Role: "user",
				Content: []ClaudeTextContent{
					{
						Type: "text",
						Text: "タイの首都は？",
					},
				},
			},
		},
	}

	// リクエストのJSONエンコード
	body, err := json.Marshal(request)
	if err != nil {
		log.Fatalf("リクエストのJSONエンコードに失敗しました: %v", err)
	}

	// モデルの呼び出し
	result, err := client.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
		Body:        body,
	})

	// エラー処理
	if err != nil {
		handleInvokeError(err, modelID)
		os.Exit(1)
	}

	// レスポンスの解析
	var response ClaudeResponse
	if err := json.Unmarshal(result.Body, &response); err != nil {
		log.Fatalf("レスポンスの解析に失敗しました: %v", err)
	}

	// 結果の表示
	fmt.Println(response.Content[0].Text)
}

// handleInvokeError はモデル呼び出し時のエラーを処理する関数
func handleInvokeError(err error, modelID string) {
	errMsg := err.Error()
	switch {
	case strings.Contains(errMsg, "no such host"):
		fmt.Printf("エラー: 選択されたリージョンでBedrockサービスが利用できません。リージョンごとのサービス提供状況を https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/ で確認してください\n")
	case strings.Contains(errMsg, "Could not resolve the foundation model"):
		fmt.Printf("エラー: モデル識別子 \"%s\" からファンデーションモデルを解決できませんでした。指定されたモデルが存在し、指定されたリージョンでアクセス可能であることを確認してください\n", modelID)
	default:
		fmt.Printf("エラー: Anthropic Claudeの呼び出しに失敗しました: %v\n", err)
	}
}
