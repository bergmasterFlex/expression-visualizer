#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Id(pub usize);

#[derive(Debug, Clone)]
pub enum ENode {
    Program {},
    Sink {
        input_anchor: super::AnchorId,
    },
    FunctionCall {
        function_declaration_id: super::FunctionDeclarationId,
        input_anchors: Vec<super::AnchorId>,
        output_anchor: super::AnchorId,
    },
    ConstDecl {
        r#type: EType,
        output_anchor: super::AnchorId,
    },
    TypeCast {
        r#type: EType,
        input_anchor: super::AnchorId,
        output_anchor: super::AnchorId,
    },
    VarDecl {
        name: String,
        r#type: EType,
        output_anchor: super::AnchorId,
    },
    Match {
        patterns: Vec<super::node::Id>,
        input_anchor: super::AnchorId,
        output_anchor: super::AnchorId,
    },
    Pattern {
        parent_match: super::node::Id,
        r#type: EType,
        output_anchor: super::AnchorId,
    },
}

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

impl ENode {
    pub fn anchors(&self) -> Vec<(super::AnchorId, super::EAnchor)> {
        match self {
            ENode::FunctionCall {
                input_anchors,
                output_anchor,
                ..
            } => input_anchors
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, anchor_id)| {
                    (
                        anchor_id,
                        super::EAnchor::Input {
                            order_num: i,
                            name: Some(format!("param{}", i)),
                        },
                    )
                })
                .chain(vec![(output_anchor.clone(), super::EAnchor::Output)])
                .collect(),
            ENode::Sink { input_anchor } => vec![(
                input_anchor.clone(),
                super::EAnchor::Input {
                    order_num: 0,
                    name: None,
                },
            )],
            ENode::ConstDecl { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::EAnchor::Output)]
            }
            ENode::TypeCast {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    super::EAnchor::Input {
                        order_num: 0,
                        name: None,
                    },
                ),
                (output_anchor.clone(), super::EAnchor::Output),
            ],
            ENode::VarDecl { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::EAnchor::Output)]
            }
            ENode::Match {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    super::EAnchor::Input {
                        order_num: 0,
                        name: None,
                    },
                ),
                (output_anchor.clone(), super::EAnchor::Output),
            ],
            ENode::Pattern { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::EAnchor::Output)]
            }
            ENode::Program { .. } => vec![],
        }
    }
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
