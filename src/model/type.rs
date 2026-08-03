#[derive(Debug, Clone)]
pub enum EType {
    Bool { value: Option<String> },
    Int { value: Option<String> },
    Float { value: Option<String> },
    String { value: Option<String> },
    Char { value: Option<String> },
    Any,
    Undefined { message: Option<String> },
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self.clone() {
            EType::Bool { value } => value.unwrap_or("bool".to_string()),
            EType::Int { value } => value.unwrap_or("int".to_string()),
            EType::Float { value } => value.unwrap_or("float".to_string()),
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
            EType::Any => "any".to_string(),
            EType::Undefined { message } => message.unwrap_or("undefined".to_string()),
        }
    }
}
