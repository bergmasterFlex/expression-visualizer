#[derive(Debug, Clone)]
pub enum EType {
    Bool { value: Option<String> },
    Int { value: Option<String> },
    String { value: Option<String> },
    Char { value: Option<String> },
    None { message: Option<String> },
}

impl std::fmt::Display for EType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self.clone() {
            EType::Bool { value } => value.unwrap_or("bool".to_string()),
            EType::Int { value } => value.unwrap_or("int".to_string()),
            EType::String { value } => {
                if let Some(value) = value {
                    format!("\"{}\"", value)
                } else {
                    "string".to_string()
                }
            }
            EType::Char { value } => {
                if let Some(value) = value {
                    format!("'{}'", value)
                } else {
                    "char".to_string()
                }
            }
            EType::None { message } => message.unwrap_or("none".to_string()),
        };
        write!(f, "{}", text)
    }
}
